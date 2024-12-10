from PIL import Image
import torch
import torch._dynamo
import torch.fx
from einops import rearrange
import math
from torch.utils.cpp_extension import load
import torchvision
from torchvision.transforms.transforms import ToTensor

roiunpool = load(name="roiunpool", sources=["roictrl/cuda_extension/roiunpool.cpp", "roictrl/cuda_extension/roiunpool.cu"], verbose=True)

def _roi_unpooling_cuda(roi_feat, rois, rois_mask, height, width, spatial_scale, aligned):
    batch_size, num_rois, channels, roi_height, roi_width = roi_feat.shape
    target_feat = torch.zeros((batch_size, num_rois, channels, height, width), device=roi_feat.device, dtype=roi_feat.dtype)
    
    roi_feat = rearrange(roi_feat, 'b n c h w -> (b n) c h w') # 20, 3, 64, 64
    rois = rearrange(rois, "b n c -> (b n) c") # 20, 4
    rois_mask = rearrange(rois_mask, "b n -> (b n)") # 2, 10
    target_feat = rearrange(target_feat, "b n c h w -> (b n) c h w")
    
    roi_feat_ = roi_feat[rois_mask==1]
    rois_ = rois[rois_mask==1] 
    
    target_feat_ = Roi_Unpool.apply(roi_feat_, rois_, spatial_scale, height, width, roi_height, roi_width, aligned)
    
    target_feat[rois_mask==1] = target_feat_
    target_feat = rearrange(target_feat, "(b n) c h w -> b n c h w", b=batch_size)
    
    return target_feat

class Roi_Unpool(torch.autograd.Function):
    @staticmethod
    def forward(ctx, roi_feat, rois, spatial_scale, height, width, pooled_height, pooled_width, aligned):

        target_feat = roiunpool.roi_unpool_forward(roi_feat, rois, spatial_scale, height, width, pooled_height, pooled_width, aligned)

        ctx.save_for_backward(rois)
        ctx.height, ctx.width, ctx.pooled_height, ctx.pooled_width, ctx.spatial_scale, ctx.aligned = height, width, pooled_height, pooled_width, spatial_scale, aligned

        return target_feat

    @staticmethod
    def backward(ctx, grad):
        rois = ctx.saved_tensors[0]
        grad_input = roiunpool.roi_unpool_backward(grad.contiguous(), rois, ctx.spatial_scale, ctx.height, ctx.width, ctx.pooled_height, ctx.pooled_width, ctx.aligned)
        return grad_input, None, None, None, None, None, None, None