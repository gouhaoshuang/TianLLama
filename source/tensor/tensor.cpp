#include "tensor/tensor.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

std::size_t data_type_size(base::DataType data_type) {
    switch (data_type) {
    case base::DataType::kDataTypeFp32:
        return sizeof(float);
    case base::DataType::kDataTypeFp16:
        return sizeof(std::uint16_t);
    case base::DataType::kDataTypeInt8:
        return sizeof(std::int8_t);
    case base::DataType::kDataTypeUnknown:
    default:
        throw std::invalid_argument(
            "Tensor requires a known DataType");
    }
}

std::size_t compute_numel(
    const std::vector<std::int64_t>& dims) {
    if (dims.empty()) {
        throw std::invalid_argument(
            "Tensor dimensions cannot be empty");
    }
    std::size_t numel = 1;
    const auto max_size = std::numeric_limits<std::size_t>::max();

    for (std::int64_t dim : dims) {
        if (dim <= 0) {
            throw std::invalid_argument(
                "Every Tensor dimension must be greater than zero");
        }
        if (static_cast<std::uintmax_t>(dim) > static_cast<std::uintmax_t>(max_size)) {
            throw std::overflow_error("Tensor dimension overflow");
        }

        const auto unsigned_dim = static_cast<std::size_t>(dim);
        if (numel > max_size / unsigned_dim) {
            throw std::overflow_error("Tensor element count overflow");
        }
        numel *= unsigned_dim;
    }

    return numel;
}

std::size_t compute_byte_size(
    std::size_t numel,
    base::DataType data_type) {
    const std::size_t element_size = data_type_size(data_type);

    if (numel > std::numeric_limits<std::size_t>::max() / element_size) {
        throw std::overflow_error("Tensor byte size overflow");
    }

    return numel * element_size;
}

} // namespace

namespace tensor {

Tensor::Tensor(
    std::vector<std::int64_t> dims,
    base::DataType data_type,
    std::shared_ptr<base::DeviceAllocator> allocator)
    : dims_(std::move(dims)),
      data_type_(data_type),
      size_(compute_numel(dims_)),
      buffer_(std::make_shared<base::Buffer>(
          compute_byte_size(size_, data_type_),
          std::move(allocator))) {

    if (buffer_->empty()) {
        throw std::bad_alloc();
    }
}

Tensor::Tensor(
    std::vector<std::int64_t> dims,
    base::DataType data_type,
    std::shared_ptr<base::Buffer> buffer,
    std::size_t byte_offset)
    : dims_(std::move(dims)),
      data_type_(data_type),
      size_(compute_numel(dims_)),
      buffer_(std::move(buffer)),
      byte_offset_(byte_offset) {}

bool Tensor::empty() const {
    return !buffer_ || buffer_->ptr() == nullptr;
}

Tensor Tensor::view(
    std::vector<int64_t> dims,
    std::size_t offset) {
    if (empty())
        throw std::invalid_argument("Cannot view an empty Tensor");

    const auto count = compute_numel(dims);

    if (offset > size_ || count > size_ - offset) {
        throw std::out_of_range("Tensor view exceeds parent range");
    }

    const auto offset_bytes = compute_byte_size(offset, data_type_);
    return Tensor(std::move(dims), data_type_, buffer_, byte_offset_ + offset_bytes);
}

bool Tensor::overlaps(const Tensor& other) const {
    if (empty() || other.empty() || buffer_ != other.buffer_)
        return false;

    return byte_offset_ < other.byte_offset_ + other.byte_size() &&
           other.byte_offset_ < byte_offset_ + byte_size();
}

std::size_t Tensor::dims_size() const {
    return dims_.size();
}

std::int64_t Tensor::dim(size_t index) const {
    return dims_.at(index);
}

bool Tensor::same_shape(const Tensor& other) const {
    return dims_ == other.dims_;
}

std::size_t Tensor::size() const {
    return size_;
}
std::size_t Tensor::byte_size() const {
    return compute_byte_size(size_, data_type_);
}
const std::vector<std::int64_t>& Tensor::dims() const {
    return dims_;
}
base::DataType Tensor::data_type() const {
    return data_type_;
}
base::DeviceType Tensor::device_type() const {
    return buffer_->device_type();
}

} // namespace tensor
