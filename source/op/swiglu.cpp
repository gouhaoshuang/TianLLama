#include "op/swiglu.h"
#include "kernel/swiglu.h"

namespace op {

base::Status SwiGLULayer::forward(
    const TensorInputs& inputs,
    const TensorOutputs& outputs,
    const base::ExecutionContext& context) const {

    if (inputs.size() != 2 || outputs.size() != 1) {
        return {base::kInvalidArgument, "SwiGLU requires two inputs and one output"};
    }
    if (!inputs[0] || !inputs[1] || !outputs[0]) {
        return {base::kInvalidArgument, "SwiGLU received a null Tensor"};
    }

    const auto& gate = *inputs[0];
    const auto& up = *inputs[1];
    auto& output = *outputs[0];

    if (gate.empty() || up.empty() || output.empty()) {
        return {base::kInvalidArgument, "SwiGLU received empty storage"};
    }
    if (gate.data_type() != base::DataType::kDataTypeFp32 ||
        up.data_type() != base::DataType::kDataTypeFp32 ||
        output.data_type() != base::DataType::kDataTypeFp32) {
        return {base::kInvalidArgument, "SwiGLU supports FP32 only"};
    }
    if (!gate.same_shape(up) || !gate.same_shape(output)) {
        return {base::kInvalidArgument, "SwiGLU shapes must match"};
    }
    const auto device = gate.device_type();
    if (up.device_type() != device || output.device_type() != device) {
        return {base::kInvalidArgument, "SwiGLU devices must match"};
    }
    if (device != base::DeviceType::kDeviceCPU &&
        device != base::DeviceType::kDeviceGPU) {
        return {base::kInvalidArgument, "SwiGLU device is unsupported"};
    }
    if (output.overlaps(gate) || output.overlaps(up)) {
        return {base::kInvalidArgument, "SwiGLU requires separate output storage"};
    }
    if (device == base::DeviceType::kDeviceCPU) {
        return kernel::swiglu_cpu(gate, up, output);
    }
    return kernel::swiglu_cuda(gate, up, output, context.stream);
}

} // namespace op
