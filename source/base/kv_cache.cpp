
#include "base/kv_cache.h"

#include <cmath>

#include <memory>
#include <stdexcept>

namespace {
std::shared_ptr<base::DeviceAllocator> checked_allocator(
    std::shared_ptr<base::DeviceAllocator> allocator) {
    if (!allocator) {
        throw std::invalid_argument("KVCache requires an allocator");
    }
    const auto device = allocator->device_type();
    if (device != base::DeviceType::kDeviceCPU &&
        device != base::DeviceType::kDeviceGPU) {
        throw std::invalid_argument("KVCache requires CPU or GPU allocator");
    }
    return allocator;
}
} // namespace

namespace base {

KVCache::KVCache(
    int64_t capacity,
    int64_t heads,
    int64_t head_dim,
    std::shared_ptr<DeviceAllocator> allocator)
    : allocator_(checked_allocator(std::move(allocator))),
      keys_(
          {capacity, heads, head_dim},
          DataType::kDataTypeFp32,
          allocator_),
      values_(
          {capacity, heads, head_dim},
          DataType::kDataTypeFp32,
          allocator_) {}

size_t KVCache::token_size() const {
    return keys_.size() / static_cast<size_t>(capacity());
}

void KVCache::check_position(int64_t position) const {
    if (position != length_ || length_ >= capacity()) {
        throw std::out_of_range("KVCache requires next position within capacity");
    }
}

// 目前只支持提前最新的key，value
tensor::Tensor KVCache::key_slot(int64_t position) {
    check_position(position);
    return keys_.view(
        {keys_.dim(1), keys_.dim(2)},
        static_cast<size_t>(position) * token_size());
}

tensor::Tensor KVCache::value_slot(int64_t position) {
    check_position(position);
    return values_.view(
        {values_.dim(1), values_.dim(2)},
        static_cast<size_t>(position) * token_size());
}

Status KVCache::commit(int64_t position,
                       const ExecutionContext& context) {

    if (position != length_ || length_ >= capacity()) {
        return {kInvalidArgument, "KVCache commit position is invalid"};
    }
    // 调用者负责完整写入 K/V，并保证后续读取的执行顺序。
    ++length_;
    return {};
}

tensor::Tensor KVCache::keys() {
    if (length_ == 0)
        throw std::logic_error("KVCache is empty");
    return keys_.view({length_, keys_.dim(1), keys_.dim(2)});
}

tensor::Tensor KVCache::values() {
    if (length_ == 0)
        throw std::logic_error("KVCache is empty");
    return values_.view({length_, values_.dim(1), values_.dim(2)});
}

} // namespace base
