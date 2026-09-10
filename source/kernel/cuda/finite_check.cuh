#pragma once

#include "base/alloc.h"
#include "tensor/tensor.h"
#include <cmath>
#include <cuda_runtime.h>
#include <exception>
#include <string>
#include <vector>

namespace kernel::small_data_detail {

inline base::Status check_finite(
    const tensor::Tensor& input,
    void* stream) {
    std::vector<float> host(input.size());

    base::CUDADeviceAllocator copier;
    copier.memcpy(
        input.ptr<float>(),
        host.data(),
        input.byte_size(),
        base::MemcpyKind::kMemcpyGPU2CPU,
        stream,
        true);

    for (float value : host) {
        if (!std::isfinite(value)) {
            return {base::kInvalidArgument, "Input must be finite"};
        }
    }
    return {};
}

} // namespace kernel::small_data_detail
