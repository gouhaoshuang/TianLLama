#pragma once
#include "tensor/tensor.h"

namespace kernel {
base::Status attention_cpu(const tensor::Tensor& q,
                           const tensor::Tensor& k,
                           const tensor::Tensor& v,
                           tensor::Tensor& output);
} // namespace kernel