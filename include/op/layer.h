#pragma once

#include "base/base.h"
#include "tensor/tensor.h"
#include "base/cuda_config.h"

#include<vector>
#include<string>
#include <cstdint>

namespace op
{



using TensorInputs = std::vector<const tensor::Tensor *>; 
using TensorOutputs = std::vector<tensor::Tensor *>; 



class Layer 
{
    
public:
    virtual ~Layer() = default;

    virtual base::Status forward(
        const TensorInputs& inputs,
        const TensorOutputs& outputs
    ) const = 0;
};


} // namespace op
