#pragma once

#include "op/layer.h"


namespace op
{


class AddLayer final  : public Layer{
public:

    base::Status forward(
        const TensorInputs& inputs,
        const TensorOutputs& outputs
    ) const override;

};
} // namespace op
