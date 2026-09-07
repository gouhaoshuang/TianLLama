#pragma once

#include "op/layer.h"

namespace op {

class AddLayer final : public Layer {
  public:
    base::Status forward(
        const TensorInputs &inputs,
        const TensorOutputs &outputs,
        const base::ExecutionContext &context = {}) const override;
};
} // namespace op
