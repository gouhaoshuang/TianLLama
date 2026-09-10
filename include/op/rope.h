#pragma once

#include "op/layer.h"
#include <cstdint>

namespace op {

class RoPELayer final : public Layer {
  public:
    explicit RoPELayer(std::int64_t position, double theta = 10000.0);

    base::Status forward(
        const TensorInputs &inputs,
        const TensorOutputs &outputs,
        const base::ExecutionContext &context = {}) const override;

  private:
    std::int64_t position_;
    double theta_;
};

} // namespace op