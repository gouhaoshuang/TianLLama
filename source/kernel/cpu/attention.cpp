#include "kernel/attention.h"
#include "kernel/softmax.h"
#include <cmath>
#include <limits>
#include <memory>

namespace kernel {

// Q:[H , D]
// K:[L , H , D]
// V:[L , H , D]

// socres:[H , L]

base::Status attention_cpu(
    const tensor::Tensor& q,
    const tensor::Tensor& k,
    const tensor::Tensor& v,
    tensor::Tensor& output) {

    const size_t L = static_cast<size_t>(k.dim(0));
    const size_t H = static_cast<size_t>(q.dim(0));
    const size_t D = static_cast<size_t>(q.dim(1));

    auto allocator = std::make_shared<base::CPUDeviceAllocator>();

    tensor::Tensor scores({q.dim(0), k.dim(0)}, base::DataType::kDataTypeFp32, allocator);
    tensor::Tensor probs({q.dim(0), k.dim(0)}, base::DataType::kDataTypeFp32, allocator);

    const double scale = 1.0 / std::sqrt(static_cast<double>(D));

    // 1. 每个 head 的当前 Q，与该 head 的所有可见 K 做点积。
    for (std::size_t h = 0; h < H; h++) {
        for (size_t s = 0; s < L; s++) {
            double dot = 0;

            for (size_t d = 0; d < D; d++) {
                dot += static_cast<double>(
                    q.ptr<float>()[h * D + d] *
                    k.ptr<float>()[(s * H * D) + h * D + d]);
            }
            const double score = dot * scale;
            if (!std::isfinite(score) ||
                std::abs(score) > std::numeric_limits<float>::max()) {
                return {base::kInvalidArgument, "Attention score exceeds FP32 range"};
            }

            scores.ptr<float>()[h * L + s] = static_cast<float>(score);
        }
    }

    // 2. 复用现有二维稳定 Softmax：每一行是一个 head。
    const auto status = softmax_cpu(scores, probs);
    if (!status)
        return status;

    for (size_t h = 0; h < H; h++) {
        for (size_t d = 0; d < D; d++) {
            double sum = 0;

            for (size_t s = 0; s < L; s++) {
                sum += static_cast<double>(probs.ptr<float>()[h * L + s] *
                                           v.ptr<float>()[s * H * D + h * D + d]);
            }
            output.ptr<float>()[h * D + d] = static_cast<float>(sum);
        }
    }
    return {};
}

} // namespace kernel
