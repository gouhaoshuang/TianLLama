#pragma once

#include "base/base.h"
#include <cuda_runtime_api.h>
#include <string>

namespace kernel::cuda_check {
inline base::Status cuda_error(cudaError_t status, const char* operation) {
    return {base::kInternalError,
            std::string(operation) + ": " + cudaGetErrorString(status)};
}

// 只检查启动后的错误状态，不等待 GPU。
// 应紧接在对应 kernel launch 后调用，避免夹入其他 CUDA 操作。
inline base::Status check_launch_error(const char* operation) {
    const auto error = cudaGetLastError();
    if (error != cudaSuccess)
        return cuda_error(error, operation);
    return {};
}

// 完成一次 launch 的检查，并按本课策略决定是否等待。
// 名称表示包装函数的处理结束，不表示显式 stream 的 GPU 计算已完成。
inline base::Status finish_launch(
    void* stream,
    const char* launch_operation,
    const char* execution_operation) {

    const auto launched = check_launch_error(launch_operation);
    if (!launched) return launched;

    if (stream == nullptr) {
        const auto error = cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
        if (error != cudaSuccess)
            return cuda_error(error, execution_operation);
    }
    return {};
}
} // namespace kernel::cuda_check
