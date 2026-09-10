#include "op/softmax.h"
#include "kernel/softmax.h"
#include <cmath>

namespace op {

base::Status SoftmaxLayer::forward(
    const TensorInputs& inputs,
    const TensorOutputs& outputs,
    const base::ExecutionContext& context) const {

    if (inputs.size() != 1 || outputs.size() != 1 || !inputs[0] || !outputs[0]) {
        return {base::kInvalidArgument, "Softmax requires one non-null input/output"};
    }

    const auto& x = *inputs[0];
    auto& y = *outputs[0];

    if (x.empty() || y.empty() || x.dims_size() != 2 || !x.same_shape(y)) {
        return {base::kInvalidArgument, "Softmax requires matching [rows, columns]"};
    }
    if (x.data_type() != base::DataType::kDataTypeFp32 ||
        y.data_type() != base::DataType::kDataTypeFp32) {
        return {base::kInvalidArgument, "Softmax supports FP32 only"};
    }
    if (x.device_type() != y.device_type()) {
        return {base::kInvalidArgument, "Softmax devices must match"};
    }
    if (x.overlaps(y)) {
        return {base::kInvalidArgument, "Softmax requires separate output storage"};
    }

    if (x.device_type() == base::DeviceType::kDeviceCPU) {
        for (std::size_t i = 0; i < x.size(); ++i) {
            if (!std::isfinite(x.ptr<float>()[i])) {
                return {base::kInvalidArgument, "Softmax input must be finite"};
            }
        }
        return kernel::softmax_cpu(x, y);
    }
    if (x.device_type() == base::DeviceType::kDeviceGPU) {
        return kernel::softmax_cuda(x, y, context.stream);
    }
    return {base::kFunctionUnImplement, "Softmax device is not supported"};
}

} // namespace op
