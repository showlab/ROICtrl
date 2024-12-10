#include <torch/extension.h>

at::Tensor roi_unpool_forward_kernel(const at::Tensor& roi_feat, const at::Tensor& rois, double spatial_scale, int64_t height, int64_t width, int64_t pooled_height, int64_t pooled_width, bool aligned);
at::Tensor roi_unpool_backward_kernel(const at::Tensor& grad, const at::Tensor& rois, double spatial_scale, int64_t height, int64_t width, int64_t pooled_height, int64_t pooled_width, bool aligned);

at::Tensor roi_unpool_forward(
    const at::Tensor& roi_feat, // Input roi feat map.
    const at::Tensor& rois, // List of ROIs.
    double spatial_scale, // The scale of the image features. ROIs will be scaled to this.
    int64_t height, // The height of the original feature map.
    int64_t width, // The width of the original feature map.
    int64_t pooled_height, // The height of the pooled feature map.
    int64_t pooled_width, // The width of the pooled feature
    bool aligned) // The flag for pixel shift
// along each axis.
{
    return roi_unpool_forward_kernel(
        roi_feat,
        rois,
        spatial_scale,
        height,
        width,
        pooled_height,
        pooled_width,
        aligned);
}

at::Tensor roi_unpool_backward(
    const at::Tensor& grad,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t height,
    int64_t width,
    int64_t pooled_height,
    int64_t pooled_width,
    bool aligned) {
    
    return roi_unpool_backward_kernel(
        grad,
        rois,
        spatial_scale,
        height,
        width,
        pooled_height,
        pooled_width,
        aligned);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("roi_unpool_forward", &roi_unpool_forward);
    m.def("roi_unpool_backward", &roi_unpool_backward);
}