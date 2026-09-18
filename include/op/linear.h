#pragma once

#include "op/layer_param.h"

namespace op {

class LinearLayer final : public LayerParam {

  public:
    explicit LinearLayer(
        std::shared_ptr<const tensor::Tensor> weight,
        std::shared_ptr<const tensor::Tensor> bias = nullptr);

    base::Status forward(
        const TensorInputs& inputs,
        const TensorOutputs& outputs,
        const base::ExecutionContext& context = {}) const override;

  private:
    std::shared_ptr<const tensor::Tensor> bias_;
};

} // namespace op
