
#include "finite_check.cuh"
#include "kernel/rope.h"
#include <cmath>
#include <cstddef>
#include <cuda_runtime.h>
#include <limits>
#include <string>

namespace {

constexpr int kThreads = 128;

base::Status cuda_error(cudaError_t status, const char* operation) {
    return {base::kInternalError,
            std::string(operation) + ": " + cudaGetErrorString(status)};
}

__global__ void rope_fp32_kernel(
    const float* input,
    float* output,
    std::size_t pairs,
    std::size_t head_dim,
    std::int64_t position,
    double theta) {

    const size_t first = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t step = gridDim.x * blockDim.x;

    const size_t pairs_per_head = head_dim / 2;

    for (size_t pair = first; pair < pairs; pair += step) {
        const size_t d = 2 * (pair % pairs_per_head);
        const double frequency = pow(theta, -static_cast<double>(d) / static_cast<double>(head_dim));

        const double angle = static_cast<double>(position) * frequency;
        const double c = cos(angle);
        const double s = sin(angle);

        const std::size_t offset = 2 * pair;

        const double a = input[offset];
        const double b = input[offset + 1];
        output[offset] = static_cast<float>(a * c - b * s);
        output[offset + 1] = static_cast<float>(a * s + b * c);
    }
}

} // namespace

namespace kernel {

base::Status rope_cuda(const tensor::Tensor& input,
                       tensor::Tensor& output,
                       std::int64_t position,
                       double theta,
                       void* stream) {

    const base::Status valid = small_data_detail::check_finite(input, stream);
    if (!valid)
        return valid;

    const size_t heads = static_cast<size_t>(input.dim(0));
    const size_t head_dim = static_cast<size_t>(input.dim(1));

    const std::size_t pairs = input.size() / 2;
    auto blocks = static_cast<unsigned int>(std::min<size_t>((pairs - 1) / kThreads + 1, 65535));

    const cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

    rope_fp32_kernel<<<blocks, kThreads, 0, cuda_stream>>>(
        input.ptr<float>(),
        output.ptr<float>(),
        pairs,
        head_dim,
        position,
        theta);

    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        return cuda_error(status, "ROPE kernel launch");
    }
    status = cudaStreamSynchronize(cuda_stream);
    if (status != cudaSuccess) {
        return cuda_error(status, "ROPE kernel execution");
    }
    return {};
}
} // namespace kernel
