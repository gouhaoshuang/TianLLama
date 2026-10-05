#include "cuda_utils.cuh"
#include "kernel/attention.h"
#include "kernel/softmax.h"

#include <cmath>

namespace {

constexpr int kThreads = 128;

__global__ void attention_scores_kernel(
    const float* q,
    const float* k,
    float* scores,
    size_t L,
    size_t Hkv,
    size_t D,
    size_t group,
    float scale) {

    const size_t h = blockIdx.x;
    const size_t tid = threadIdx.x;
    const size_t kv_h = h / group;

    // 每个线程处理一个或多个历史位置 t。
    for (size_t t = tid; t < L; t += kThreads) {
        double dot = 0.0;
        for (size_t d = 0; d < D; d++) {
            dot += q[h * D + d] * k[t * (Hkv * D) + kv_h * D + d];
        }
        scores[h * L + t] = dot * scale;
    }
}
__global__ void attention_weighted_sum_kernel(
    const float* probs,
    const float* v,
    float* output,
    size_t L,
    size_t Hkv,
    size_t D,
    size_t group) {

    const size_t h = blockIdx.x;
    const size_t tid = threadIdx.x;
    const size_t kv_h = h / group;

    for (size_t d = tid; d < D; d += kThreads) {
        double dot = 0.0;
        for (size_t l = 0; l < L; l++) {
            dot += probs[h * L + l] * v[l * (Hkv * D) + kv_h * D + d];
        }
        output[h * D + d] = static_cast<float>(dot);
    }
}

} // namespace

namespace kernel {
// Q:[H_q , D]
// K:[L , H_kv , D]
// V:[L , H_kv , D]

// socres:[H_q , L]
base::Status attention_cuda(const tensor::Tensor& q,
                            const tensor::Tensor& k,
                            const tensor::Tensor& v,
                            tensor::Tensor& output,
                            void* stream) {

    const size_t L = static_cast<size_t>(k.dim(0));
    const size_t Hq = static_cast<size_t>(q.dim(0));
    const size_t Hkv = static_cast<size_t>(k.dim(1));
    const size_t D = static_cast<size_t>(q.dim(1));

    const size_t group = Hq / Hkv; // Layer 已检查整除且非零。

    const auto blocks = static_cast<unsigned int>(Hq);
    const double scale = 1.0 / std::sqrt(static_cast<double>(D));

    auto allocator = std::make_shared<base::CUDADeviceAllocator>();

    tensor::Tensor scores({q.dim(0), k.dim(0)}, base::DataType::kDataTypeFp32, allocator);
    tensor::Tensor probs({q.dim(0), k.dim(0)}, base::DataType::kDataTypeFp32, allocator);

    const cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

    attention_scores_kernel<<<blocks, kThreads, 0, cuda_stream>>>(
        q.ptr<float>(),
        k.ptr<float>(),
        scores.ptr<float>(),
        L,
        Hkv,
        D,
        group,
        scale);

    auto error = cudaGetLastError();
    if (error != cudaSuccess) return cuda_check::cuda_error(error, "Attention scores launch");

    const auto status = softmax_cuda(scores, probs, stream);
    if (!status) return status;

    attention_weighted_sum_kernel<<<blocks, kThreads, 0, cuda_stream>>>(
        probs.ptr<float>(),
        v.ptr<float>(),
        output.ptr<float>(),
        L,
        Hkv,
        D,
        group);
    error = cudaGetLastError();

    if (error != cudaSuccess)
        return cuda_check::cuda_error(error, "Attention weighted sum launch");

    // 等待最后一个 kernel 完成，函数返回后临时 probs 才可以释放。
    error = cudaStreamSynchronize(cuda_stream);
    if (error != cudaSuccess)
        return cuda_check::cuda_error(error, "Attention execution");

    return {};
}
} // namespace kernel
