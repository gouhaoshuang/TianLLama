#include "op/rope.h"
#include "kernel/rope.h"
#include <cmath>
#include <stdexcept>

namespace op {

RoPELayer::RoPELayer(std::int64_t position, double theta, base::RopeLayout layout)
    : position_(position), theta_(theta), layout_(layout) {

    if (position < 0 || !std::isfinite(theta) || theta <= 1.0) {
        throw std::invalid_argument("RoPE requires position >= 0 and finite theta > 1");
    }
    if (layout != base::RopeLayout::kInterleaved &&
        layout != base::RopeLayout::kHalfSplit) {
        throw std::invalid_argument("Invalid RoPE layout");
    }
}

base::Status RoPELayer::forward(
    const TensorInputs& inputs,
    const TensorOutputs& outputs,
    const base::ExecutionContext& context) const {

    if (inputs.size() != 1 || outputs.size() != 1 ||
        !inputs[0] || !outputs[0]) {
        return {base::kInvalidArgument, "RoPE requires one non-null input/output"};
    }

    const tensor::Tensor& x = *inputs[0];
    tensor::Tensor& y = *outputs[0];

    if (x.empty() ||
        y.empty() ||
        x.dims_size() != 2 || // x，y 的是二维tensor
        !x.same_shape(y)) {
        return {base::kInvalidArgument, "RoPE requires matching [heads, head_dim]"};
    }

    if (x.dim(1) % 2 != 0) {
        return {base::kInvalidArgument, "RoPE head_dim must be even"};
    }

    if (x.data_type() != base::DataType::kDataTypeFp32 ||
        y.data_type() != base::DataType::kDataTypeFp32) {
        return {base::kInvalidArgument, "RoPE supports FP32 only"};
    }

    if (x.device_type() != y.device_type()) {
        return {base::kInvalidArgument, "RoPE devices must match"};
    }

    if (x.overlaps(y)) {
        return {base::kInvalidArgument, "RoPE requires separate output storage"};
    }

    if (x.device_type() == base::DeviceType::kDeviceCPU) {
        for (std::size_t i = 0; i < x.size(); ++i) {
            if (!std::isfinite(x.ptr<float>()[i])) {
                return {base::kInvalidArgument, "RoPE input must be finite"};
            }
        }
        return kernel::rope_cpu(x, y, position_, theta_, layout_);
    }

    if (x.device_type() == base::DeviceType::kDeviceGPU) {
        return kernel::rope_cuda(x, y, position_, theta_, context.stream, layout_);
    }
    return {base::kFunctionUnImplement, "RoPE device is not supported"};
}

} // namespace op