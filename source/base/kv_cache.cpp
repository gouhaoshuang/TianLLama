
#include "base/kv_cache.h"

#include <cmath>

#include <memory>
#include <stdexcept>

namespace base {
KVCache::KVCache(
    int64_t capacity,
    int64_t heads,
    int64_t head_dim)
    : keys_(
          {capacity, heads, head_dim},
          DataType::kDataTypeFp32,
          std::make_shared<CPUDeviceAllocator>()),
      values_(
          {capacity, heads, head_dim},
          DataType::kDataTypeFp32,
          std::make_shared<CPUDeviceAllocator>()) {}

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

Status KVCache::commit(int64_t position) {
    if (position != length_ || length_ >= capacity()) {
        return {kInvalidArgument, "KVCache commit position is invalid"};
    }
    const auto offset = static_cast<size_t>(position) * token_size();

    // 前提：调用者已经完整写入两个槽位；不能拿此检查替代初始化。
    for (std::size_t i = 0; i < token_size(); ++i) {
        if (!std::isfinite(keys_.ptr<float>()[offset + i]) ||
            !std::isfinite(values_.ptr<float>()[offset + i])) {
            return {kInvalidArgument, "KVCache requires finite K/V"};
        }
    }
    ++length_; // 只有 K/V 均已写入、检查通过，才让本 token 可见。
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
