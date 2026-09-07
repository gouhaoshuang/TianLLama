#pragma once

#include "base/base.h"
#include "tensor/tensor.h"

namespace kernel {

base::Status rmsnorm_cpu(
    const tensor::Tensor &input,
    const tensor::Tensor &weight,
    tensor::Tensor &output,
    float epsilon);

base::Status rmsnorm_cuda(
    const tensor::Tensor &input,
    const tensor::Tensor &weight,
    tensor::Tensor &output,
    float epsilon,
    void *stream = nullptr);
} // namespace kernel
