#include "op/attention.h"

#include "kernel/attention.h"
#include <cmath>
#include <stdexcept>
namespace op {

AttentionLayer::AttentionLayer(
    int64_t heads,
    int64_t capacity,
    std::shared_ptr<base::DeviceAllocator> allocator) {

    if (heads <= 0 || capacity <= 0 || !allocator) {
        throw std::invalid_argument("Invalid Attention workspace configuration");
    }

    const auto device = allocator->device_type();

    if (device != base::DeviceType::kDeviceCPU &&
        device != base::DeviceType::kDeviceGPU)
        throw std::invalid_argument("Unsupported Attention workspace device");

    scores_storage_ = std::make_unique<tensor::Tensor>(
        std::vector<int64_t>{heads, capacity},
        base::DataType::kDataTypeFp32,
        allocator);
    probs_storage_ = std::make_unique<tensor::Tensor>(
        std::vector<int64_t>{heads, capacity},
        base::DataType::kDataTypeFp32,
        allocator);
}

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
    const auto device = q.device_type();

    for (const tensor::Tensor* t : {&q, &k, &v, static_cast<const tensor::Tensor*>(&y)}) {
        if (t->empty() || t->data_type() != base::DataType::kDataTypeFp32) {
            return {base::kInvalidArgument, "Attention requires non-empty FP32 tensors"};
        }
        if (t->device_type() != device) {
            return {base::kInvalidArgument, "Attention Tensor devices must match"};
        }
    }
    if (device != base::DeviceType::kDeviceCPU &&
        device != base::DeviceType::kDeviceGPU) {
        return {base::kFunctionUnImplement, "Attention device is not supported"};
    }

    if (q.dims_size() != 2 || k.dims_size() != 3 ||
        !k.same_shape(v) || !q.same_shape(y)) {
        return {base::kInvalidArgument, "Attention requires Q/Y[Hq,D], K/V[L,Hkv,D]"};
    }

    // q:[H_q , D] , K / V :[L , H_kv , D]
    if (q.dim(0) <= 0 || k.dim(1) <= 0 ||
        q.dim(1) != k.dim(2) ||
        q.dim(0) % k.dim(1) != 0) {
        return {base::kInvalidArgument, "Attention requires Hq divisible by Hkv and equal D"};
    }

    if (y.overlaps(q) || y.overlaps(k) || y.overlaps(v)) {
        return {base::kInvalidArgument, "Attention output must not overlap inputs"};
    }

    const int64_t Hq = q.dim(0);
    const int64_t L = k.dim(0);
    if (!scores_storage_ || !probs_storage_)
        return {base::kInvalidArgument, "Attention workspace is not configured"};
    if (scores_storage_->device_type() != device ||
        probs_storage_->device_type() != device)
        return {base::kInvalidArgument, "Attention workspace device mismatch"};
    if (scores_storage_->dim(0) != Hq || probs_storage_->dim(0) != Hq ||
        L > scores_storage_->dim(1) || L > probs_storage_->dim(1))
        return {base::kInvalidArgument, "Attention workspace shape/capacity mismatch"};
    for (const tensor::Tensor* t : {&q, &k, &v, static_cast<const tensor::Tensor*>(&y)}) {
        if (scores_storage_->overlaps(*t) || probs_storage_->overlaps(*t))
            return {base::kInvalidArgument, "Attention workspace overlaps input/output"};
    }

    auto scores = scores_storage_->view({Hq, L});
    auto probs = probs_storage_->view({Hq, L});
    if (device == base::DeviceType::kDeviceGPU)
        return kernel::attention_cuda(q, k, v, y, scores, probs, context.stream);
    return kernel::attention_cpu(q, k, v, y, scores, probs);
}

} // namespace op
