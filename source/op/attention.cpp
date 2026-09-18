#include "op/attention.h"

#include "kernel/attention.h"
#include <cmath>

namespace op {

base::Status AttentionLayer::forward(
    const TensorInputs& inputs,
    const TensorOutputs& outputs,
    const base::ExecutionContext& context) const {
    if (inputs.size() != 3 || outputs.size() != 1 ||
        !inputs[0] || !inputs[1] || !inputs[2] || !outputs[0]) {
        return {base::kInvalidArgument, "Attention requires Q/K/V and one output"};
    }

    const auto& q = *inputs[0];
    const auto& k = *inputs[1];
    const auto& v = *inputs[2];
    auto& y = *outputs[0];

    for (const tensor::Tensor* t : {&q, &k, &v, static_cast<const tensor::Tensor*>(&y)}) {
        if (t->empty() || t->data_type() != base::DataType::kDataTypeFp32) {
            return {base::kInvalidArgument, "Attention requires non-empty FP32 tensors"};
        }
        if (t->device_type() != base::DeviceType::kDeviceCPU) {
            return {base::kFunctionUnImplement, "Attention currently supports CPU only"};
        }
    }

    if (q.dims_size() != 2 || k.dims_size() != 3 ||
        !k.same_shape(v) || !q.same_shape(y)) {
        return {base::kInvalidArgument, "Attention requires Q/Y[Hq,D], K/V[L,Hkv,D]"};
    }
    // q:[H_q , D] , K / V :[L , H_kv , D]
    if (q.dim(0) <= 0 || k.dim(1) <= 0 || q.dim(1) != k.dim(2) ||
        q.dim(0) % k.dim(1) != 0) {
        return {base::kInvalidArgument, "Attention requires Hq divisible by Hkv and equal D"};
    }

    if (y.overlaps(q) || y.overlaps(k) || y.overlaps(v)) {
        return {base::kInvalidArgument, "Attention output must not overlap inputs"};
    }
    for (const tensor::Tensor* t : {&q, &k, &v}) {
        for (std::size_t i = 0; i < t->size(); ++i) {
            if (!std::isfinite(t->ptr<float>()[i])) {
                return {base::kInvalidArgument, "Attention inputs must be finite"};
            }
        }
    }
    return kernel::attention_cpu(q, k, v, y);
}

} // namespace op
