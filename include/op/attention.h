#pragma once

#include "op/layer.h"

namespace op {

class AttentionLayer final : public Layer {

  public:
    base::Status forward(
        const TensorInputs& inputs,
        const TensorOutputs& outputs,
        const base::ExecutionContext& context = {}) const;
};

} // namespace op
