#include "sampler/greedy_sampler.h"
#include "kernel/argmax.h"

#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>

namespace sampler {
GreedySampler::GreedySampler(base::DeviceType device)
    : device_(device) {
    if (device == base::DeviceType::kDeviceCPU) {
        return;
    }
    if (device_ == base::DeviceType::kDeviceGPU) {
        result_ = base::Buffer(sizeof(kernel::ArgmaxResult),
                               std::make_shared<base::CUDADeviceAllocator>());
        return; 
    }
    throw std::invalid_argument("Unsupported sampler device");
}
base::Status GreedySampler::sample(
    const tensor::Tensor& logits,
    int32_t& token_id,
    void* stream) {
    if (logits.empty() || logits.data_type() != base::DataType::kDataTypeFp32 ||
        logits.device_type() != device_ || logits.dims_size() != 2 ||
        logits.dim(0) != 1 || logits.dim(1) <= 0 ||
        logits.dim(1) > std::numeric_limits<int32_t>::max())
        return {base::kInvalidArgument, "Sampler requires matching-device FP32 logits [1,V]"};
    if (device_ == base::DeviceType::kDeviceGPU) {
        return kernel::argmax_cuda(
            logits.ptr<float>(),
            logits.size(),
            static_cast<kernel::ArgmaxResult*>(result_.ptr()),
            token_id,
            stream);
    }
    // 走到这里一定是 CPU 模式，且上面已经确认输入也是 CPU Tensor。
    // 直接读取 CPU 数据，不调用 CUDA，也不进行 H2D / D2H 复制。

    const float* values = logits.ptr<float>();
    int32_t best = 0;
    for (size_t i = 0; i < logits.size(); i++) {
        if (!std::isfinite(values[i]))
            return {base::kInternalError, "Non-finite logits"};
        if (values[i] > values[best])
            best = static_cast<int32_t>(i);
    }
    token_id = best; // 成功才修改调用者的输出。
    return {};
}

} // namespace sampler
