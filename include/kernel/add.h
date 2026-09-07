#pragma once

#include "base/base.h"
#include "tensor/tensor.h"



namespace kernel
{

base::Status add_cpu(
    const tensor::Tensor& left,
    const tensor::Tensor& right,
    tensor::Tensor&  output
) ;

base::Status add_cuda(
    const tensor::Tensor& left,
    const tensor::Tensor& right,
    tensor::Tensor&  output,
    void* stream = nullptr
) ;


}  // namespace kernel
