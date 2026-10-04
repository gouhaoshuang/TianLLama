
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

    tensor::Tensor k = key_slot(position);
    tensor::Tensor v = value_slot(position);
    const std::size_t count = token_size();

    const float* k_data = nullptr;
    const float* v_data = nullptr;

    std::vector<float> host_k;
    std::vector<float> host_v;

    if (device_type() == DeviceType::kDeviceCPU) {
        k_data = k.ptr<float>();
        v_data = v.ptr<float>();
    } else if (device_type() == DeviceType::kDeviceGPU) {
        host_k.resize(count);
        host_v.resize(count);

        allocator_->memcpy(k.ptr<float>(),
                           host_k.data(),
                           k.byte_size(),
                           MemcpyKind::kMemcpyGPU2CPU,
                           context.stream,
                           true);
        allocator_->memcpy(v.ptr<float>(),
                           host_v.data(),
                           v.byte_size(),
                           MemcpyKind::kMemcpyGPU2CPU,
                           context.stream,
                           true);
        k_data = host_k.data();
        v_data = host_v.data();
    }

    for (size_t i = 0; i < count; i++) {
        if (!std::isfinite(k_data[i]) || !std::isfinite(v_data[i])) {
            return {kInvalidArgument, "KVCache requires finite K/V"};
        }
    }
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
