#pragma once

#include "base/base.h"
#include "base/execution_context.h"
#include "tensor/tensor.h"

#include <cstdint>
#include <string>
#include <vector>

namespace op {

using TensorInputs = std::vector<const tensor::Tensor *>;
using TensorOutputs = std::vector<tensor::Tensor *>;

class Layer {

  public:
    virtual ~Layer() = default;

    virtual base::Status forward(
        const TensorInputs &inputs,
        const TensorOutputs &outputs,
        const base::ExecutionContext &context = {}) const = 0;
};

} // namespace op
