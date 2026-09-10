#pragma once

#include "tensor/tensor.h"

namespace kernel {

base::Status softmax_cpu(
    const tensor::Tensor& input,
    tensor::Tensor& output);

base::Status softmax_cuda(
    const tensor::Tensor& input,
    tensor::Tensor& output,
    void *stream = nullptr
);
} // namespace kernel
