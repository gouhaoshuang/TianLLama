#include "kernel/linear.h"

#include <cstddef>
#include <cub/block/block_reduce.cuh>
#include <cuda_runtime_api.h>
#include <limits>
#include <string>

namespace {

const int kThreads = 128;

base::Status cuda_error(cudaError_t status, const char *operation) {
    return {
        base::kInternalError,
        std::string(operation) + ": " + cudaGetErrorString(status)
    };
}
__global__ void linear_fp32_kernel(
    const float *x,
    const float *w,
    float *y,
    size_t K,
    size_t N) {

    const size_t index = blockIdx.x;
    const size_t m = index / N;
    const size_t n = index % N;
    const size_t tid = threadIdx.x;

    float local_sum = 0.0F;
    for (size_t k = tid; k < K; k += kThreads) {
        local_sum += x[m * K + k] * w[n * K + k];
    }

    using Reduce = cub::BlockReduce<float, kThreads>;
    __shared__ Reduce::TempStorage storage;

    const float sum = Reduce(storage).Sum(local_sum);
    if(tid == 0){
        y[index] = sum;
    }
}

} // namespace

namespace kernel {

// X: [M , K] , W: [N , K] , Y: [M , N]
base::Status linear_cuda(
    const tensor::Tensor &input,
    const tensor::Tensor &weight,
    tensor::Tensor &output,
    void *stream) {

    const size_t K = static_cast<size_t>(input.dim(1));
    const size_t N = static_cast<size_t>(weight.dim(0));
    const size_t blocks = output.size();

    // 与项目当前 RMSNorm 一样，对一维 launch 数量做上界检查。
    if (blocks == 0 || blocks > static_cast<std::size_t>(
                                    std::numeric_limits<int>::max())) {
        return {base::kInvalidArgument, "Linear output count exceeds launch limits"};
    }

    const cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    linear_fp32_kernel<<<
        static_cast<unsigned int>(blocks),
        kThreads,
        0,
        cuda_stream>>>(input.ptr<float>(), weight.ptr<float>(), output.ptr<float>(), K, N);

    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        return cuda_error(status, "Linear kernel launch");
    }

    status = cudaStreamSynchronize(cuda_stream);
    if (status != cudaSuccess) {
        return cuda_error(status, "Linear kernel execution");
    }
    return {};
}

} // namespace kernel
