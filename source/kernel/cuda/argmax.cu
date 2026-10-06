#include "cuda_utils.cuh"
#include "kernel/argmax.h"

#include <cub/block/block_reduce.cuh>
#include <cuda_runtime.h>
#include <limits>
#include <math_constants.h>

namespace

{
const int kThreads = 128;
struct Candidate {
    float value;
    int32_t id;
    int32_t invalid;
};

struct Better {
    __device__ Candidate operator()(Candidate a, Candidate b) const {
        const bool choose_b = b.value > a.value ||
                              (b.value == a.value && b.id < a.id);
        Candidate result = choose_b ? b : a;
        result.invalid = a.invalid | b.invalid;
        return result;
    }
};

__global__ void argmax_fp32_kernel(
    const float* input,
    size_t count,
    kernel::ArgmaxResult* output) {

    using Reduce = cub::BlockReduce<Candidate, kThreads>;
    __shared__ Reduce::TempStorage storage;

    Candidate local{-CUDART_INF_F, INT32_MAX, 0};

    for (size_t i = threadIdx.x; i < count; i += blockDim.x) {
        const float value = input[i];

        if (!isfinite(value)) {
            local.invalid = 1;
        } else {
            local = Better{}(local, Candidate{value, static_cast<int32_t>(i), 0});
        }
    }
    const Candidate best = Reduce(storage).Reduce(local, Better{});
    if (threadIdx.x == 0) {
        *output = kernel::ArgmaxResult{best.id, best.invalid};
    }
}
} // namespace

namespace kernel {
base::Status argmax_cuda(
    const float* input,
    size_t count,
    ArgmaxResult* device_result,
    int32_t& token_id,
    void* stream) {

    if (!input || !device_result || count == 0 ||
        count > static_cast<std::size_t>(std::numeric_limits<int32_t>::max()))
        return {base::kInvalidArgument, "Invalid GPU argmax arguments"};

    const auto cuda_stream = static_cast<cudaStream_t>(stream);

    argmax_fp32_kernel<<<1, kThreads, 0, cuda_stream>>>(
        input,
        count,
        device_result);

    auto error = cudaGetLastError();
    if (error != cudaSuccess) {
        return cuda_check::cuda_error(error, "Argmax launch");
    }

    ArgmaxResult host{};
    error = cudaMemcpyAsync(
        &host,
        device_result,
        sizeof(host),
        cudaMemcpyDeviceToHost,
        cuda_stream);
    if (error != cudaSuccess)
        return cuda_check::cuda_error(error, "Argmax result copy");
    // 同一 stream 内，复制排在 kernel 后；返回前等待复制与执行完成。
    error = cudaStreamSynchronize(cuda_stream);
    if (error != cudaSuccess)
        return cuda_check::cuda_error(error, "Argmax execution");
    if (host.invalid)
        return {base::kInternalError, "Non-finite logits"};
    if (host.token_id < 0 || static_cast<std::size_t>(host.token_id) >= count)
        return {base::kInternalError, "Invalid argmax result"};
    token_id = host.token_id;
    return {};
}
} // namespace kernel
