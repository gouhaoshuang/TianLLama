#include "base/utils.h"
#include "base/alloc.h"
#include <bit>
#include <cmath>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

namespace base {

tensor::Tensor read_f32(const std::filesystem::path& path,
                        const std::vector<int64_t>& shape,
                        std::uint64_t offset_bytes) {

    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
                  "This reader requires IEEE 754 FP32");

    static_assert(std::endian::native == std::endian::little,
                  "This reader currently supports little-endian hosts only");

    std::ifstream file(path, std::ios::binary);

    if (!file) throw std::runtime_error("Cannot open: " + path.string());

    tensor::Tensor result(shape, DataType::kDataTypeFp32, std::make_shared<CPUDeviceAllocator>());

    const size_t bytes = result.byte_size();
    const auto file_bytes = std::filesystem::file_size(path);

    if (offset_bytes % sizeof(float) != 0 || offset_bytes > file_bytes ||
        bytes > file_bytes - offset_bytes) {
        throw std::runtime_error("Invalid FP32 range: " + path.string());
    }

    if (!std::in_range<std::streamoff>(offset_bytes) ||
        !std::in_range<std::streamsize>(bytes)) {
        throw std::overflow_error("File range exceeds stream limits");
    }

    // 把读取位置移动到“文件开头加 offset 字节”
    file.seekg(static_cast<std::streamoff>(offset_bytes), std::ios::beg);
    if (!file)
        throw std::runtime_error("Cannot seek: " + path.string());

    if (!file.read(reinterpret_cast<char*>(result.ptr<float>()),
                   static_cast<std::streamsize>(bytes)))
        throw std::runtime_error("Short read: " + path.string());

    for (std::size_t i = 0; i < result.size(); ++i) {
        if (!std::isfinite(result.ptr<float>()[i]))
            throw std::runtime_error("Non-finite tensor: " + path.string());
    }
    return result;
}

// 整文件接口：复用核心实现，再要求没有多余字节。
tensor::Tensor read_f32(const std::filesystem::path& path,
                        const std::vector<std::int64_t>& shape) {
    auto result = read_f32(path, shape, 0);
    if (std::filesystem::file_size(path) != result.byte_size())
        throw std::runtime_error("File size mismatch: " + path.string());
    return result;
}

} // namespace base
