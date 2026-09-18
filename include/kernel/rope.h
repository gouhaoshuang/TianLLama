#pragma once
#include "op/rope.h"
#include "tensor/tensor.h"
#include <cstdint>

namespace kernel {
// 内部接口：由 Layer 保证 CPU/FP32、二维、偶数 head_dim、独立输出。
base::Status rope_cpu(const tensor::Tensor& input,
                      tensor::Tensor& output,
                      std::int64_t position,
                      double theta,
                      base::RopeLayout layout = base::RopeLayout::kInterleaved);

base::Status rope_cuda(const tensor::Tensor& input,
                       tensor::Tensor& output,
                       std::int64_t position,
                       double theta,
                       void* stream = nullptr,
                       base::RopeLayout layout = base::RopeLayout::kInterleaved);
} // namespace kernel
