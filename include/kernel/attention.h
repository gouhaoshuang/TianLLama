#pragma once
#include "tensor/tensor.h"

namespace kernel {

base::Status attention_cpu(const tensor::Tensor& q,
                           const tensor::Tensor& k,
                           const tensor::Tensor& v,
                           tensor::Tensor& output);

base::Status attention_cuda(const tensor::Tensor& q,
                            const tensor::Tensor& k,
                            const tensor::Tensor& v,
                            tensor::Tensor& output,
                            void* stream = nullptr);

} // namespace kernel