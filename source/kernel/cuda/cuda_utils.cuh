#pragma once

#include "base/base.h"
#include <cuda_runtime_api.h>
#include <string>

namespace kernel::cuda_check {
inline base::Status cuda_error(cudaError_t status, const char* operation) {
    return {base::kInternalError,
            std::string(operation) + ": " + cudaGetErrorString(status)};
}

} // namespace kernel::cuda_check
