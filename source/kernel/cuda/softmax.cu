#include "finite_check.cuh"
#include "kernel/softmax.h"
#include <cstddef>
#include <cub/block/block_reduce.cuh>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <limits>
#include <math_constants.h>

namespace {
constexpr int kThreads = 128;

base::Status cuda_error(cudaError_t status, const char* operation) {
    return {base::kInternalError,
            std::string(operation) + ": " + cudaGetErrorString(status)};
}

__global__ void softmax_fp32_kernel(
    const float* input,
    float* output,
    std::size_t columns) {

    const size_t offset = (blockIdx.x) * columns;
    const size_t tid = threadIdx.x;

    using Reduce = cub::BlockReduce<float, kThreads>;
    __shared__ Reduce::TempStorage storage;

    __shared__ float row_max;
    __shared__ float row_sum;

    float local_max = -CUDART_INF_F;

    for (size_t j = tid; j < columns; j += blockDim.x) {
        local_max = fmaxf(local_max, input[offset + j]);
    }

    const float maximum = Reduce(storage).Reduce(local_max, cub::Max());

    if (tid == 0)
        row_max = maximum;
    // 广播 row_max；同时保证后面的 Sum 可以安全复用 storage。
    __syncthreads();

    float local_sum = 0.0F;

    for (std::size_t j = tid; j < columns; j += blockDim.x) {
        const float value = expf(input[offset + j] - row_max);
        output[offset + j] = value; // 独立输出暂存指数，不额外申请 Tensor。
        local_sum += value;
    }

    const float total = Reduce(storage).Sum(local_sum);
    if (tid == 0)
        row_sum = total;
    __syncthreads(); // 所有线程都能读到 thread 0 写入的 row_sum。

    for (std::size_t j = tid; j < columns; j += blockDim.x) {
        output[offset + j] /= row_sum;
    }
}

} // namespace

namespace kernel {

base::Status softmax_cuda(
    const tensor::Tensor& input,
    tensor::Tensor& output,
    void* stream) {

    const auto rows = static_cast<unsigned int>(input.dim(0));
    const auto clos = static_cast<size_t>(input.dim(1));
    if (rows == 0 ||
        rows > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        return {base::kInvalidArgument, "Softmax row count exceeds launch limits"};
    }
    const auto valid = small_data_detail::check_finite(input, stream);
    if (!valid)
        return valid;
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    softmax_fp32_kernel<<<rows, kThreads, 0, cuda_stream>>>(
        input.ptr<float>(),
        output.ptr<float>(),
        clos);

    auto status = cudaGetLastError();
    if (status != cudaSuccess) {
        return cuda_error(status, "softmax kernel launch");
    }
    status = cudaStreamSynchronize(cuda_stream);
    if (status != cudaSuccess) {
        return cuda_error(status, "softmax kernel execution");
    }
}

} // namespace kernel
