#include "cuda_utils.cuh"
#include "kernel/swiglu.h"
#include <cstddef>
#include <cuda_runtime.h>
#include <limits>
#include <string>

namespace {

constexpr int kThreads = 128;


__global__ void swiglu_fp32_kernel(
    const float* gate,
    const float* up,
    float* output,
    size_t count) {

    const std::size_t i = (blockIdx.x) * blockDim.x + threadIdx.x;

    if (i < count) {
        const float v = gate[i];
        const float e = expf(-fabsf(v));

        const float sigmoid = v >= 0.0F ? 1.0F / (1.0F + e) : e / (1.0F + e);
        output[i] = (v * sigmoid) * up[i];
    }
}

} // namespace

namespace kernel {
base::Status swiglu_cuda(const tensor::Tensor& gate,
                         const tensor::Tensor& up,
                         tensor::Tensor& output,
                         void* stream) {

    const std::size_t count = output.size();
    if (count == 0) {
        return {base::kInvalidArgument, "SwiGLU output is empty"};
    }

    const std::size_t blocks = (count - 1) / kThreads + 1;

    if (blocks > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        return {base::kInvalidArgument, "SwiGLU output exceeds launch limits"};
    }

    const auto cuda_stream = static_cast<cudaStream_t>(stream);

    swiglu_fp32_kernel<<<static_cast<unsigned int>(blocks), kThreads, 0, cuda_stream>>>(
        gate.ptr<float>(),
        up.ptr<float>(),
        output.ptr<float>(),
        count);

    return cuda_check::finish_launch(
        stream,
        "SwiGLU kernel launch",
        "SwiGLU kernel execution");
}
} // namespace kernel