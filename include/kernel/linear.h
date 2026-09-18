#pragma once

#include "base/base.h"
#include "tensor/tensor.h"

namespace kernel {

// 内部计算接口：调用前由 LinearLayer 完成 shape/dtype/device/别名检查。
base::Status linear_cpu(
    const tensor::Tensor& input,
    const tensor::Tensor& weight,
    tensor::Tensor& output,
    const tensor::Tensor* bias = nullptr);

base::Status linear_cuda(
    const tensor::Tensor& input,
    const tensor::Tensor& weight,
    tensor::Tensor& output,
    void* stream = nullptr,
    const tensor::Tensor* bias = nullptr);
} // namespace kernel
