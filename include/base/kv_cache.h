#pragma once

#include "base/execution_context.h"
#include "tensor/tensor.h"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace base {

class KVCache {

  public:
    KVCache(
        int64_t capacity, // KVCache 设计容量
        int64_t heads,
        int64_t head_dim,
        std::shared_ptr<DeviceAllocator> allocator = std::make_shared<CPUDeviceAllocator>());

    int64_t length() const { return length_; }
    int64_t capacity() const { return keys_.dim(0); }
    DeviceType device_type() const { return allocator_->device_type(); }

    tensor::Tensor key_slot(int64_t position);
    tensor::Tensor value_slot(int64_t position);

    Status commit(int64_t position,
                  const ExecutionContext& context = {});

    tensor::Tensor keys(); // 只返回 [length, H, D] 有效前缀。
    tensor::Tensor values();

    void reset() { length_ = 0; }

  private:
    void check_position(int64_t position) const;

    size_t token_size() const; // 一个token 的维度，也就是model_size.

    std::shared_ptr<DeviceAllocator> allocator_;
    tensor::Tensor keys_;
    tensor::Tensor values_;

    std::int64_t length_ = 0; // 当前 KVCache 存储的KV 的长度。也就是最后一个KV 的编号.
};
} // namespace base
