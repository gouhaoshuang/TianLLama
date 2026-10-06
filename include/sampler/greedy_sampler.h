#pragma once
#include "base/buffer.h"
#include "tensor/tensor.h"
#include <cstdint>

namespace sampler {

class GreedySampler {

  public:
    explicit GreedySampler(base::DeviceType device);

    base::Status sample(
        const tensor::Tensor& logits,
        int32_t& token_id,
        void* stream = nullptr);

  private:
    base::DeviceType device_;
    base::Buffer result_; // CPU 模式为空；GPU 模式持有 8 字节。
};

} // namespace sampler
