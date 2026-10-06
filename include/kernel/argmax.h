#pragma once
#include "base/base.h"
#include <cstddef>
#include <cstdint>

namespace kernel {

struct ArgmaxResult {
    int32_t token_id;
    int32_t invalid; // 任意原始 logit 为 NaN/Inf，则为 1。
};

static_assert(sizeof(ArgmaxResult) == 8);

base::Status argmax_cuda(const float* input,
                         size_t count,
                         ArgmaxResult* device_result,
                         int32_t& token_id,
                         void* stream = nullptr);

} // namespace kernel
