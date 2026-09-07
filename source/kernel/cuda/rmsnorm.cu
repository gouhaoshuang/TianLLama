#include "kernel/rmsnorm.h"

#include <cub/block/block_reduce.cuh>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <limits>
#include <string>

namespace {
const int kThreads = 128;

base::Status cuda_error(cudaError_t status, const char *operation) {
    return {
        base::kInternalError,
        std::string(operation) + ": " + cudaGetErrorString(status)};
}

__global__ void rmsnorm_fp32_kernel(
    const float *input,
    const float *weight,
    float *output,
    size_t dim,
    float epsilon) {
    const size_t row = blockIdx.x;
    const size_t offset = row * dim;
    const size_t tid = threadIdx.x;

    float local_sum = 0.0F;

    for (size_t j = tid; j < dim; j += blockDim.x) {
        const float value = input[offset + j];
        local_sum += value * value;
    }
    using Reduce = cub::BlockReduce<float, kThreads>;
    __shared__ Reduce::TempStorage storage;
    __shared__ float scale;

    const float sum = Reduce(storage).Sum(local_sum);
    if (tid == 0) {
        scale = rsqrtf(sum / static_cast<float>(dim) + epsilon);
    }

    __syncthreads();
    for (size_t j = tid; j < dim; j += blockDim.x) {
        output[offset + j] = input[offset + j] * scale * weight[j];
    }
}

} // namespace

namespace kernel {

base::Status rmsnorm_cuda(
    const tensor::Tensor &input,
    const tensor::Tensor &weight,
    tensor::Tensor &output,
    float epsilon,
    void *stream) {

    const size_t dim = weight.size();
    const size_t rows = input.size() / dim;

    // 当前 NVIDIA GPU 的一维 grid.x 上限为 2^31-1。
    if (rows == 0 || rows > static_cast<std::size_t>(
                                std::numeric_limits<int>::max())) {
        return {base::kInvalidArgument, "RMSNorm row count exceeds launch limits"};
    }

    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    rmsnorm_fp32_kernel<<<
        static_cast<unsigned int>(rows),
        kThreads,
        0,
        cuda_stream>>>(input.ptr<float>(), weight.ptr<float>(), output.ptr<float>(), dim, epsilon);

    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        return cuda_error(status, "RMSNorm kernel launch");
    }

    status = cudaStreamSynchronize(cuda_stream);
    if (status != cudaSuccess) {
        return cuda_error(status, "RMSNorm kernel execution");
    }
    return {};
}

} // namespace kernel
