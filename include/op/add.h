#pragma once

#include "op/layer.h"


namespace op
{


class AddLayer : public Layer{
public:

    virtual base::Status forward(
        const TensorInputs& inputs,
        const TensorOutputs& outputs
    ) const override;

};

} // namespace op
