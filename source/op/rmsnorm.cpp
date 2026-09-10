#include "op/rmsnorm.h"
#include "kernel/rmsnorm.h"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace op {

RmsNormLayer::RmsNormLayer(
    std::shared_ptr<const tensor::Tensor> weight,
    float epsilon)
    : LayerParam(weight), epsilon_(epsilon) {

    if (!std::isfinite(epsilon_) || epsilon_ <= 0.0F) {
        throw std::invalid_argument("RMSNorm epsilon must be finite and positive");
    }
    const tensor::Tensor& w = this->weight();

    if (w.empty() || w.dims_size() != 1 ||
        w.data_type() != base::DataType::kDataTypeFp32) {
        throw std::invalid_argument("RMSNorm weight must be a non-empty FP32 [D] Tensor");
    }
    if (w.device_type() != base::DeviceType::kDeviceCPU &&
        w.device_type() != base::DeviceType::kDeviceGPU) {
        throw std::invalid_argument("RMSNorm weight device is unsupported");
    }
}

base::Status RmsNormLayer::forward(
    const TensorInputs& inputs,
    const TensorOutputs& outputs,
    const base::ExecutionContext& context) const {
    if (inputs.size() != 1 || outputs.size() != 1) {
        return {base::kInvalidArgument, "RMSNorm requires one input and one output"};
    }

    if (!inputs[0] || !outputs[0]) {
        return {base::kInvalidArgument, "RMSNorm received a null Tensor"};
    }

    const tensor::Tensor& x = *inputs[0];
    tensor::Tensor& y = *outputs[0];
    const tensor::Tensor& w = this->weight();

    if (x.empty() || y.empty() || w.empty() ||
        x.dims_size() == 0 || w.dims_size() != 1) {
        return {base::kInvalidArgument, "RMSNorm received empty memory or invalid rank"};
    }

    if (x.data_type() != base::DataType::kDataTypeFp32 ||
        y.data_type() != base::DataType::kDataTypeFp32 ||
        w.data_type() != base::DataType::kDataTypeFp32) {
        return {base::kInvalidArgument, "RMSNorm supports FP32 only"};
    }

    if (!x.same_shape(y) || x.dim(x.dims_size() - 1) != w.dim(0)) {
        return {base::kInvalidArgument, "RMSNorm requires x/y [..., D] and weight [D]"};
    }

    const base::DeviceType device = x.device_type();
    if (y.device_type() != device || w.device_type() != device) {
        return {base::kInvalidArgument, "RMSNorm Tensor devices must match"};
    }

    if (device != base::DeviceType::kDeviceCPU &&
        device != base::DeviceType::kDeviceGPU) {
        return {base::kInvalidArgument, "RMSNorm device is unsupported"};
    }
    if (y.overlaps(x) || y.overlaps(w)) {
        return {base::kInvalidArgument, "RMSNorm requires separate output storage"};
    }

    if (device == base::DeviceType::kDeviceCPU) {
        return kernel::rmsnorm_cpu(x, w, y, epsilon_);
    }
    return kernel::rmsnorm_cuda(x, w, y, epsilon_, context.stream);
}

} // namespace op
