#pragma once

#include "op/layer_param.h"

namespace op {

class RmsNormLayer final : public LayerParam {

  public:
    explicit RmsNormLayer(
        std::shared_ptr<const tensor::Tensor> weight,
        float epsilon = 1e-5F);

    base::Status forward(
        const TensorInputs &inputs,
        const TensorOutputs &outputs,
        const base::ExecutionContext& context = {}) const override;

  private:
    float epsilon_;
};

} // namespace op
