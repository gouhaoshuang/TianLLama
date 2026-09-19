#pragma once
#include "tensor/tensor.h"
#include <filesystem>
#include <vector>

namespace base {

tensor::Tensor read_f32(const std::filesystem::path& file,
                        const std::vector<int64_t>& shape);

tensor::Tensor read_f32(const std::filesystem::path& file,
                        const std::vector<int64_t>& shape,
                        std::uint64_t offset_bytes);
} // namespace base
