#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/native/cuda/KernelUtils.cuh>

#include "cuda_helpers.h"


__device__ bool is_legal_coord(int y, int x, int height, int width) {
    if (y >= 0 && y < height && x >= 0 && x < width) {
      return true;
    } 
    else {
      return false;
    }
}

template <typename T>
__device__ T bilinear_interpolate(
    const T* offset_roi_feat_,
    int pooled_height,
    int pooled_width,
    T y,
    T x,
    int index /* index for debug only*/) {

  int y_low = (int)y;
  int x_low = (int)x;
  int y_high = (int)ceil(y);
  int x_high = (int)ceil(x);

  T ly = y - y_low;
  T lx = x - x_low;
  T hy = 1. - ly, hx = 1. - lx;

  T w1 = 0, w2 = 0, w3 = 0, w4 = 0;
  T v1 = 0, v2 = 0, v3 = 0, v4 = 0;

  if (is_legal_coord(y_low, x_low, pooled_height, pooled_width)) {
    w1 = hy * hx;
    v1 = offset_roi_feat_[y_low * pooled_width + x_low];
  }
  else{
    w1 = 0;
    v1 = 0;
  }

  if (is_legal_coord(y_low, x_high, pooled_height, pooled_width)) {
    w2 = hy * lx;
    v2 = offset_roi_feat_[y_low * pooled_width + x_high];
  }
  else{
    w2 = 0;
    v2 = 0;
  }

  if (is_legal_coord(y_high, x_low, pooled_height, pooled_width)) {
    w3 = ly * hx;
    v3 = offset_roi_feat_[y_high * pooled_width + x_low];
  }
  else{
    w3 = 0;
    v3 = 0;
  }

  if (is_legal_coord(y_high, x_high, pooled_height, pooled_width)) {
    w4 = ly * lx;
    v4 = offset_roi_feat_[y_high * pooled_width + x_high];
  }
  else{
    w4 = 0;
    v4 = 0;
  }

  T val;
  if (1 - (w1 + w2 + w3 + w4) < 0.00001) {
    val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  }
  else {
    val = 0;
  }
  
  return val;
}


template <typename T>
__global__ void roi_unpool_forward_kernel_impl(
    int nthreads,
    const T* roi_feat_,
    const T spatial_scale,
    int channels,
    int height,
    int width,
    int pooled_height,
    int pooled_width,
    bool aligned,
    const T* rois,
    T* output) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, h, w) is an element in the final output
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;

    const T* offset_rois = rois + n * 4;

    // Do not using rounding; this implementation detail is critical
    T offset = aligned ? (T)0.5 : (T)0.0;
    T feat_start_w = int(ceil(offset_rois[0] * spatial_scale - offset));
    T feat_start_h = int(ceil(offset_rois[1] * spatial_scale - offset));
    T feat_end_w = int(ceil(offset_rois[2] * spatial_scale - offset));
    T feat_end_h = int(ceil(offset_rois[3] * spatial_scale - offset));

    T feat_width = feat_end_w - feat_start_w;
    T feat_height = feat_end_h - feat_start_h;

    if (!aligned) {
      // Force malformed ROIs to be 1x1
      feat_width = max(feat_width, (T)1.);
      feat_height = max(feat_height, (T)1.);
    }

    if (h >= feat_start_h && h < feat_end_h && w >= feat_start_w && w < feat_end_w) {
      T roi_y = ((h - feat_start_h) / feat_height) * pooled_height;
      T roi_x = ((w - feat_start_w) / feat_width) * pooled_width;
      const T* offset_roi_feat_ = roi_feat_ + (n * channels + c) * pooled_height * pooled_width;
      output[index] = bilinear_interpolate(offset_roi_feat_, pooled_height, pooled_width, roi_y, roi_x, index);
    }
  }
}


at::Tensor roi_unpool_forward_kernel(
  const at::Tensor& roi_feat, 
  const at::Tensor& rois, 
  double spatial_scale, 
  int64_t height, 
  int64_t width, 
  int64_t pooled_height, 
  int64_t pooled_width, 
  bool aligned) {
  
  TORCH_CHECK(roi_feat.is_cuda(), "roi_feat must be a CUDA tensor");
  TORCH_CHECK(rois.is_cuda(), "rois must be a CUDA tensor");

  at::cuda::CUDAGuard device_guard(roi_feat.device());

  auto batch_size = roi_feat.size(0);
  auto channels = roi_feat.size(1);

  at::Tensor output = at::zeros(
      {batch_size, channels, height, width}, roi_feat.options());
  auto output_size = batch_size * channels * height * width;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(std::min(
      ceil_div(static_cast<int64_t>(output_size), static_cast<int64_t>(512)),
      static_cast<int64_t>(4096)));  
  dim3 block(512);

  if (output.numel() == 0) {
    AT_CUDA_CHECK(cudaGetLastError());
    return output;
  }

  auto roi_feat_ = roi_feat.contiguous(), rois_ = rois.contiguous();
  // printf("batchsize: %d channels: %d height: %d width: %d", batch_size, channels, height, width);
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      roi_feat.scalar_type(), "roi_unpool_forward_kernel", [&] {
        roi_unpool_forward_kernel_impl<scalar_t><<<grid, block, 0, stream>>>(
            output_size,
            roi_feat_.data_ptr<scalar_t>(),
            spatial_scale,
            channels,
            height,
            width,
            pooled_height,
            pooled_width,
            aligned,
            rois_.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>());
      });
  AT_CUDA_CHECK(cudaGetLastError());
  return output;
}


template <typename T>
__device__ void bilinear_interpolate_gradient(
    int pooled_height,
    int pooled_width,
    T y,
    T x,
    T& w1,
    T& w2,
    T& w3,
    T& w4,
    int& x_low,
    int& x_high,
    int& y_low,
    int& y_high,
    int index) {
  // deal with cases that inverse elements are out of feature map boundary
  y_low = (int)y;
  x_low = (int)x;
  y_high = (int)ceil(y);
  x_high = (int)ceil(x);

  T ly = y - y_low;
  T lx = x - x_low;
  T hy = 1. - ly, hx = 1. - lx;

  if (is_legal_coord(y_low, x_low, pooled_height, pooled_width)) {
    w1 = hy * hx;
  }
  else{
    w1 = 0;
  }

  if (is_legal_coord(y_low, x_high, pooled_height, pooled_width)) {
    w2 = hy * lx;
  }
  else{
    w2 = 0;
  }

  if (is_legal_coord(y_high, x_low, pooled_height, pooled_width)) {
    w3 = ly * hx;
  }
  else{
    w3 = 0;
  }

  if (is_legal_coord(y_high, x_high, pooled_height, pooled_width)) {
    w4 = ly * lx;
  }
  else{
    w4 = 0;
  }
}


template <typename T>
__global__ void roi_unpool_backward_kernel_impl(
    int nthreads,
    const T* grad_output,
    const T spatial_scale,
    int channels,
    int height,
    int width,
    int pooled_height,
    int pooled_width,
    bool aligned,
    T* grad_input,
    const T* rois,
    int n_stride,
    int c_stride,
    int h_stride,
    int w_stride,
    const int memory_span) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {

    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;

    const T* offset_rois = rois + n * 4;

    // Do not using rounding; this implementation detail is critical
    T offset = aligned ? (T)0.5 : (T)0.0;
    T feat_start_w = int(ceil(offset_rois[0] * spatial_scale - offset));
    T feat_start_h = int(ceil(offset_rois[1] * spatial_scale - offset));
    T feat_end_w = int(ceil(offset_rois[2] * spatial_scale - offset));
    T feat_end_h = int(ceil(offset_rois[3] * spatial_scale - offset));

    T feat_width = feat_end_w - feat_start_w;
    T feat_height = feat_end_h - feat_start_h;

    if (!aligned) {
      // Force malformed ROIs to be 1x1
      feat_width = max(feat_width, (T)1.);
      feat_height = max(feat_height, (T)1.);
    }

    if (h >= feat_start_h && h < feat_end_h && w >= feat_start_w && w < feat_end_w) {
      // step1: index the gradient using the tensor strides to access the correct values.
      const int output_offset = n * n_stride + c * c_stride;
      const T* offset_grad_output = grad_output + output_offset;
      // grad at this point
      
      // input offset to save grad
      const int input_offset = (n * channels + c) * pooled_height * pooled_width;

      T roi_y = ((h - feat_start_h) / feat_height) * pooled_height;
      T roi_x = ((w - feat_start_w) / feat_width) * pooled_width;
      
      T w1, w2, w3, w4;
      int x_low, x_high, y_low, y_high;

      bilinear_interpolate_gradient(
        pooled_height, pooled_width, roi_y, roi_x,
        w1, w2, w3, w4,
        x_low, x_high, y_low, y_high, index);

      T g1 = offset_grad_output[h * h_stride + w * w_stride] * w1;
      T g2 = offset_grad_output[h * h_stride + w * w_stride] * w2;
      T g3 = offset_grad_output[h * h_stride + w * w_stride] * w3;
      T g4 = offset_grad_output[h * h_stride + w * w_stride] * w4;

      if (1 - (w1 + w2 + w3 + w4) < 0.00001) { // remove boundary 
        if (w1 != 0) {
          at::native::fastAtomicAdd(
            grad_input,
            input_offset + y_low * pooled_width + x_low,
            memory_span,
            static_cast<T>(g1),
            true);
        }

        if (w2 != 0) {
          at::native::fastAtomicAdd(
            grad_input,
            input_offset + y_low * pooled_width + x_high,
            memory_span,
            static_cast<T>(g2),
            true);
        }

        if (w3 != 0) {
          at::native::fastAtomicAdd(
            grad_input,
            input_offset + y_high * pooled_width + x_low,
            memory_span,
            static_cast<T>(g3),
            true);
        }

        if (w4 != 0) {
          at::native::fastAtomicAdd(
            grad_input,
            input_offset + y_high * pooled_width + x_high,
            memory_span,
            static_cast<T>(g4),
            true);
        }
      }
    }
  }
}


at::Tensor roi_unpool_backward_kernel(
    const at::Tensor& grad,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t height,
    int64_t width,
    int64_t pooled_height,
    int64_t pooled_width,
    bool aligned) {
    
    TORCH_CHECK(grad.is_cuda(), "grad must be a CUDA tensor");
    TORCH_CHECK(rois.is_cuda(), "rois must be a CUDA tensor");

    auto batch_size = grad.size(0);
    auto channels = grad.size(1);
    
    at::cuda::CUDAGuard device_guard(grad.device());
    at::Tensor grad_input =
      at::zeros({batch_size, channels, pooled_height, pooled_width}, grad.options());

    // b: 2, c: 3, pool height: 64, pool width: 64
    // printf("b: %d, c: %d, pool height: %d, pool width: %d\n", batch_size, channels, pooled_height, pooled_width);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    dim3 grid(std::min(
      ceil_div(static_cast<int64_t>(grad.numel()), static_cast<int64_t>(512)),
      static_cast<int64_t>(4096)));
    dim3 block(512);

    // printf("grad:%d", grad.numel()); // 983040 (2, 3, 320, 512)

    if (grad.numel() == 0) {
      AT_CUDA_CHECK(cudaGetLastError());
      return grad_input;
    }

    int n_stride = grad.stride(0);
    int c_stride = grad.stride(1);
    int h_stride = grad.stride(2);
    int w_stride = grad.stride(3);
    // n_stride:491520 (3*320*512), c_stride:163840 (320*512), h_stride:512, w_stride:1
    // printf("n_stride:%d, c_stride:%d, h_stride:%d, w_stride:%d", n_stride, c_stride, h_stride, w_stride);
    
    auto rois_ = rois.contiguous();
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad.scalar_type(), "roi_align_backward_kernel", [&] {
        roi_unpool_backward_kernel_impl<scalar_t><<<grid, block, 0, stream>>>(
            grad.numel(),
            grad.data_ptr<scalar_t>(),
            spatial_scale,
            channels,
            height,
            width,
            pooled_height,
            pooled_width,
            aligned,
            grad_input.data_ptr<scalar_t>(),
            rois_.data_ptr<scalar_t>(),
            n_stride,
            c_stride,
            h_stride,
            w_stride,
            grad_input.numel());
      });
    AT_CUDA_CHECK(cudaGetLastError());
    
    return grad_input;

}