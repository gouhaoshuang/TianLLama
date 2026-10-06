#pragma once

#include "base/base.h"

#include <cstdio>
#include <cuda_runtime_api.h>
#include <stdexcept>
#include <string>

namespace base {

class CudaStream {

  public:
    explicit CudaStream(bool enabled) {
        if (!enabled) {
            return; // CPU 模型不创建 CUDA stream
        }
        const auto error = cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
        if (error != cudaSuccess) {
            throw std::runtime_error(
                std::string("Create model stream: ") + cudaGetErrorString(error));
        }
    }
    ~CudaStream() noexcept {
        if (!stream_) return;
        // 资源所有者须先等待，再释放 Tensor；这里仅销毁 stream。
        const auto error = cudaStreamDestroy(stream_);
        if (error != cudaSuccess) {
            std::fprintf(stderr, "Destroy model stream: %s\n", cudaGetErrorString(error));
        }
    }
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    void* get() const { return static_cast<void*>(stream_); }

    Status synchronize() {
        if (!stream_) return {};

        const auto error = cudaStreamSynchronize(stream_);
        if (error != cudaSuccess) {
            return {kInternalError,
                    std::string("Model stream execution: ") +
                        cudaGetErrorString(error)};
        }
        return {};
    }

  private:
    cudaStream_t stream_ = nullptr;
};

} // namespace base



