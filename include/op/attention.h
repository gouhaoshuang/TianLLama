#pragma once

#include "op/layer.h"
#include <cstdint>
#include <memory>

namespace op {

class AttentionLayer final : public Layer {

  public:
    AttentionLayer() = default;

    AttentionLayer(
        int64_t heads,
        int64_t capacity,
        std::shared_ptr<base::DeviceAllocator> allocator);

    base::Status forward(
        const TensorInputs& inputs,
        const TensorOutputs& outputs,
        const base::ExecutionContext& context = {}) const;

  private:
    std::unique_ptr<tensor::Tensor> scores_storage_;
    std::unique_ptr<tensor::Tensor> probs_storage_;
};

} // namespace op
