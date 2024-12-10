import torch
import torch.nn as nn
import torch.utils.checkpoint
import torch.nn.functional as F

from diffusers.models.attention_processor import Attention
from diffusers.models.attention import FeedForward
import math
import torch
import torch.distributed
from torch import nn
from einops import rearrange
import torchvision
from roictrl.cuda_extension.roiunpool import _roi_unpooling_cuda
from copy import deepcopy


def get_fourier_embeds_from_boundingbox(embed_dim, box):
    """
    Args:
        embed_dim: int
        box: a 3-D tensor [B x N x 4] representing the bounding boxes for GLIGEN pipeline
    Returns:
        [B x N x embed_dim] tensor of positional embeddings
    """

    batch_size, num_boxes = box.shape[:2]

    emb = 100 ** (torch.arange(embed_dim) / embed_dim)
    emb = emb[None, None, None].to(device=box.device, dtype=box.dtype)
    emb = emb * box.unsqueeze(-1)

    emb = torch.stack((emb.sin(), emb.cos()), dim=-1)
    emb = emb.permute(0, 1, 3, 4, 2).reshape(batch_size, num_boxes, embed_dim * 2 * 4)

    return emb


class BoundingboxProjection(nn.Module):
    def __init__(self, fourier_freqs=8):
        super().__init__()

        self.fourier_embedder_dim = fourier_freqs
        self.position_dim = fourier_freqs * 2 * 4  # 2: sin/cos, 4: xyxy

        self.linears = nn.Sequential(
                nn.Linear(self.position_dim, 512),
                nn.SiLU(),
                nn.Linear(512, 512),
                nn.SiLU(),
                nn.Linear(512, self.position_dim),
            )
        self.null_position_feature = torch.nn.Parameter(torch.zeros([self.position_dim]))

    def forward(
        self,
        boxes,
        masks
    ):
        masks = masks.unsqueeze(-1)

        # embedding position (it may includes padding as placeholder)
        xyxy_embedding = get_fourier_embeds_from_boundingbox(self.fourier_embedder_dim, boxes)  # B*N*4 -> B*N*C

        # learnable null embedding
        xyxy_null = self.null_position_feature.view(1, 1, -1)

        # replace padding with learnable null embedding
        xyxy_embedding = xyxy_embedding * masks + (1 - masks) * xyxy_null
        objs = self.linears(xyxy_embedding)
        return objs


roi_resolution_multi_scale = {
    8: (25, 25),
    16: (19, 19),
    32: (13, 13),
    64: (7, 7),
}


def roi_align(feature_map, rois, spatial_scale, original_size, output_size):
    """
    Args:
        feature_map (Tensor): Tensor of shape (N, C, H, W) containing the original feature map.
        rois (Tensor): Tensor of shape (K, 4) containing the ROI coordinates (x1, y1, x2, y2).
        output_size (tuple): The size of the output feature map (height, width).

    Returns:
        Tensor: ROI pooled features of shape (K, C, output_height, output_width).
    """
    feat_size = deepcopy(original_size)

    feat_size[0] = int(feat_size[0] * spatial_scale)
    feat_size[1] = int(feat_size[1] * spatial_scale)
    
    feature_map = rearrange(feature_map, "b (h w) c -> b c h w", h=feat_size[0], w=feat_size[1])
    rois = [roi * torch.tensor([original_size[1], original_size[0], original_size[1], original_size[0]], dtype=roi.dtype, device=roi.device) for roi in rois]
    
    roi_features = torchvision.ops.roi_align(feature_map, rois, spatial_scale=spatial_scale, output_size=output_size, sampling_ratio=2, aligned=True)
    roi_features = rearrange(roi_features, "(b n) c h w -> b n c h w", b=len(rois))
    return roi_features


def roi_unpooling(roi_features, rois, rois_masks, spatial_scale, original_size):
    """
    Args:
        rois (Tensor): Tensor of shape (N, 4) containing the ROI coordinates.
        roi_features (Tensor): Tensor of shape (N, C, H, W) containing the ROI features.
        output_size (tuple): The size of the output feature map (height, width).
        original_size (tuple): The size of the original feature map (height, width).

    Returns:
        Tensor: The original feature map with the ROI features reassembled.
    """
    feat_size = deepcopy(original_size)

    feat_size[0] = int(feat_size[0] * spatial_scale)
    feat_size[1] = int(feat_size[1] * spatial_scale)
    
    rois = rois * torch.tensor([original_size[1], original_size[0], original_size[1], original_size[0]], dtype=rois.dtype, device=rois.device)

    target_feat = _roi_unpooling_cuda(roi_features, rois, rois_masks, feat_size[0], feat_size[1], spatial_scale, aligned=True)
    return target_feat


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None

        for pool_type in self.pool_types:
            if pool_type=='avg':
                fg_mask = x == 0
                avg_pool = x.sum(dim=[2,3]) / fg_mask.sum(dim=[2,3])
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type=='max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                max_pool = max_pool.view(max_pool.shape[0], -1)
                channel_att_raw = self.mlp(max_pool)
            else:
                raise NotImplementedError

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid(channel_att_raw).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


class FuseWeightGenerator(nn.Module):
    def __init__(self, in_channels, num_rois):
        super(FuseWeightGenerator, self).__init__()
        self.channel_reduction = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        self.channel_gate = ChannelGate(num_rois+1, 1, ['avg', 'max'])
        self.num_rois = num_rois

    def get_attn_reg(self):
        fg_mask, fuse_weight = self.fg_mask, self.fuse_weight.squeeze(2)

        fg_mask = torch.any(fg_mask[:,1:,:,:]!=0, dim=1)
        fuse_weight = fuse_weight[:, 0, :, :]
        global_attn_reg = (fuse_weight * fg_mask).sum()/fg_mask.sum()
        return global_attn_reg

    def forward(self, x):
        batch, _, _, _, _ = x.shape
        fg_mask = x.sum(2) != 0
        
        x_reshaped = rearrange(x, "b n c h w -> (b n) c h w")
        
        fuse_weight = self.channel_reduction(x_reshaped)
        fuse_weight = rearrange(fuse_weight, "(b n) c h w -> b (n c) h w", b=batch)

        # shuffle roi
        perm = torch.randperm(self.num_rois)
        global_feat, roi_feat = torch.split(fuse_weight, [1, self.num_rois], dim=1)
        shuffled_roi_feat = roi_feat[:, perm, :, :]
        
        shuffled_fuse_weight = self.channel_gate(torch.cat([global_feat, shuffled_roi_feat], dim=1))
        global_feat, shuffled_roi_feat = torch.split(shuffled_fuse_weight, [1, self.num_rois], dim=1)

        inv_perm = torch.argsort(perm)
        roi_feat = shuffled_roi_feat[:, inv_perm, :, :]
        fuse_weight = torch.cat([global_feat, roi_feat], dim=1)
        fuse_weight[~fg_mask] = torch.finfo(fuse_weight.dtype).min
        fuse_weight = fuse_weight.softmax(dim=1).unsqueeze(2)
        
        x = torch.sum(x * fuse_weight, dim=1)
        
        self.fuse_weight = fuse_weight
        self.fg_mask = fg_mask

        return x


class ROIFuser(nn.Module):
    def __init__(self, query_dim, cross_attention_dim, num_attention_heads, attention_head_dim, roi_scale, attention_type):
        super().__init__()
        self.roi_scale = roi_scale

        # learnable roi self-attention
        self.norm1 = nn.LayerNorm(query_dim)
        self.roi_self_attn = Attention(query_dim=query_dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=0.2
        )
        self.register_parameter("alpha_roi_self_attn", nn.Parameter(torch.tensor(0.0)))

        # roi_fuse
        self.roiattn_fuser = FuseWeightGenerator(in_channels=query_dim, num_rois=30)

        # learnable global self-attention
        self.norm2 = nn.LayerNorm(query_dim)
        self.context_proj = nn.Linear(64, query_dim)
        self.coord_self_attn = Attention(query_dim=query_dim, heads=num_attention_heads, dim_head=attention_head_dim, dropout=0.2)
        self.register_parameter("alpha_coord_self_attn", nn.Parameter(torch.tensor(0.0)))

        # learnable ffn
        self.norm3 = nn.LayerNorm(query_dim)
        self.ff = FeedForward(query_dim, activation_fn="geglu", dropout=0.2)
        self.register_parameter("alpha_dense", nn.Parameter(torch.tensor(0.0)))
        
        self.enabled = True

    def learned_fuser(self, roi_attn_output, attn_output):
        return self.roiattn_fuser(torch.cat([attn_output, roi_attn_output], dim=1))

    def roi_cross_attention_pretrained(self, pretrained_attn, roi_hidden_states, instance_embeddings, instance_masks):
        batch_size, num_roi, channel, roi_size_h, roi_size_w = roi_hidden_states.shape        
        roi_hidden_states = rearrange(roi_hidden_states, "b n c h w -> (b n) (h w) c")
        if len(instance_embeddings.shape) == 5:
            instance_embeddings = rearrange(instance_embeddings, "b n m l c -> (b n) m l c")
        else:
            instance_embeddings = rearrange(instance_embeddings, "b n l c -> (b n) l c")        
        instance_masks = rearrange(instance_masks, "b n -> (b n)").bool()
        
        roi_attn_output = torch.zeros_like(roi_hidden_states)
        roi_attn_output[instance_masks] = pretrained_attn(roi_hidden_states[instance_masks], instance_embeddings[instance_masks]).to(dtype=roi_attn_output.dtype)
        
        roi_attn_output = rearrange(roi_attn_output, "(b n) (h w) c -> b n c h w", b=batch_size, h=roi_size_h)
        return roi_attn_output

    def coord_self_attention_learned(self, global_attn_output, position_feat):
        position_feat = self.context_proj(position_feat)
        n_visual = global_attn_output.shape[1]
        global_attn_output = self.coord_self_attn(self.norm2(torch.cat([global_attn_output, position_feat], dim=1)))[:, :n_visual, :]
        return global_attn_output

    def roi_self_attention_learned(self, roi_hidden_states, instance_masks):
        batch_size, num_roi, channel, roi_size_h, roi_size_w = roi_hidden_states.shape            
        
        roi_hidden_states = rearrange(roi_hidden_states, "b n c h w -> (b n) (h w) c")
        instance_masks = rearrange(instance_masks, "b n -> (b n)").bool()

        roi_attn_output = torch.zeros_like(roi_hidden_states)
        roi_hidden_states = self.norm1(roi_hidden_states)
        roi_attn_output[instance_masks] = self.roi_self_attn(roi_hidden_states[instance_masks]).to(dtype=roi_attn_output.dtype)
        roi_attn_output = rearrange(roi_attn_output, "(b n) (h w) c -> b n c h w", b=batch_size, h=roi_size_h)
        
        return roi_attn_output

    def forward(self, normed_hidden_states, hidden_states, attn_output, pretrained_attn, instance_boxes, instance_masks, instance_embeddings, spatial_size, position_feat):
        if not self.enabled:
            return attn_output + hidden_states

        downsample_rate = int(math.sqrt((int(spatial_size[0]) * int(spatial_size[1])) / (normed_hidden_states.shape[1])))
        spatial_scale = 1 / downsample_rate
        original_size = [spatial_size[0], spatial_size[1]] # [320, 512]
        
        # step 1: get roi feature
        normed_roi_hidden_states = roi_align(normed_hidden_states, instance_boxes, spatial_scale=spatial_scale, original_size=original_size, output_size=self.roi_scale[downsample_rate])
        
        # step 2-1: pretrained cross-attn on ROI
        roi_attn_output = self.roi_cross_attention_pretrained(pretrained_attn, normed_roi_hidden_states, instance_embeddings, instance_masks)
        
        # step 2-2: roi self-attention
        roi_attn_output = roi_attn_output + self.alpha_roi_self_attn.tanh() * self.roi_self_attention_learned(roi_attn_output, instance_masks)
        
        # step 3-1: map to original size
        roi_attn_output = roi_unpooling(roi_attn_output, instance_boxes.to(dtype=roi_attn_output.dtype), instance_masks, spatial_scale=spatial_scale, original_size=original_size)        
        attn_output = rearrange(attn_output,  " b (h w) c -> b 1 c h w", h=original_size[0]//downsample_rate)

        # step 3-2: rewrite instance feature to attn_output
        final_attn_output = self.learned_fuser(roi_attn_output, attn_output)
        final_attn_output = rearrange(final_attn_output, "b c h w -> b (h w) c")
        hidden_states = hidden_states + final_attn_output

        # step 4-1: self-attention
        hidden_states = hidden_states + self.alpha_coord_self_attn.tanh() * self.coord_self_attention_learned(hidden_states, position_feat)
        hidden_states = hidden_states + self.alpha_dense.tanh() * self.ff(self.norm3(hidden_states))
        
        return hidden_states