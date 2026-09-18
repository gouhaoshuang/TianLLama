#include "op/linear.h"
#include "kernel/linear.h"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace op {

LinearLayer::LinearLayer(
    std::shared_ptr<const tensor::Tensor> weight,
    std::shared_ptr<const tensor::Tensor> bias)
    : LayerParam(weight), bias_(std::move(bias)) {

    const tensor::Tensor& w = this->weight();
    if (w.empty() || w.dims_size() != 2 ||
        w.data_type() != base::DataType::kDataTypeFp32) {
        throw std::invalid_argument("Linear weight must be non-empty FP32 [N,K]");
    }
    if (w.device_type() != base::DeviceType::kDeviceCPU &&
        w.device_type() != base::DeviceType::kDeviceGPU) {
        throw std::invalid_argument("Linear weight device is unsupported");
    }

    if (bias_) {
        if (bias_->empty() || bias_->dims_size() != 1 ||
            bias_->dim(0) != w.dim(0) ||
            bias_->data_type() != base::DataType::kDataTypeFp32)
            throw std::invalid_argument("Linear bias must be non-empty FP32 [N]");
        if (bias_->device_type() != w.device_type())
            throw std::invalid_argument("Linear bias and weight devices must match");
    }
}

base::Status LinearLayer::forward(
    const TensorInputs& inputs,
    const TensorOutputs& outputs,
    const base::ExecutionContext& context) const {

    if (inputs.size() != 1 || outputs.size() != 1 ||
        !inputs[0] || !outputs[0]) {
        return {base::kInvalidArgument, "Linear requires one input and one output"};
    }

    const tensor::Tensor& x = *inputs[0];
    tensor::Tensor& y = *outputs[0];
    const tensor::Tensor& w = this->weight();

    if (x.empty() || y.empty() || w.empty()) {
        return {base::kInvalidArgument, "Linear received empty storage"};
    }
    if (x.dims_size() != 2 || w.dims_size() != 2 || y.dims_size() != 2) {
        return {base::kInvalidArgument, "Linear requires X[M,K], W[N,K], Y[M,N]"};
    }
    if (x.data_type() != base::DataType::kDataTypeFp32 ||
        w.data_type() != base::DataType::kDataTypeFp32 ||
        y.data_type() != base::DataType::kDataTypeFp32) {
        return {base::kInvalidArgument, "Linear supports FP32 only"};
    }

    const auto device = x.device_type();
    if (w.device_type() != device || y.device_type() != device) {
        return {base::kInvalidArgument, "Linear Tensor devices must match"};
    }
    if (device != base::DeviceType::kDeviceCPU &&
        device != base::DeviceType::kDeviceGPU) {
        return {base::kInvalidArgument, "RMSNorm device is unsupported"};
    }
    // X: [M , K] , W: [N , K] , Y: [M , N]
    if (x.dim(1) != w.dim(1) ||
        y.dim(0) != x.dim(0) ||
        y.dim(1) != w.dim(0)) {
        return {base::kInvalidArgument, "Linear shape mismatch: X[M,K], W[N,K], Y[M,N]"};
    }
    if (y.overlaps(x) || y.overlaps(w)) {
        return {base::kInvalidArgument, "Linear requires separate output storage"};
    }
    if (bias_) {
        // bias: [N , 1]
        if (bias_->empty() || bias_->dims_size() != 1 ||
            bias_->dim(0) != w.dim(0) ||
            bias_->data_type() != base::DataType::kDataTypeFp32 ||
            bias_->device_type() != device)
            return {base::kInvalidArgument, "Invalid Linear bias"};
        if (y.overlaps(*bias_))
            return {base::kInvalidArgument, "Linear output overlaps bias"};
    }

    if (device == base::DeviceType::kDeviceCPU) {
        return kernel::linear_cpu(x, w, y , bias_.get());
    }
    return kernel::linear_cuda(x, w, y, context.stream , bias_.get());
}

} // namespace op
