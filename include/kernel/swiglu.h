#pragma once
#include "tensor/tensor.h"

namespace kernel {

// 内部接口：shape、dtype、device、输出别名由 Layer 检查。
base::Status swiglu_cpu(const tensor::Tensor &gate,
                        const tensor::Tensor &up,
                        tensor::Tensor &output);
                        
base::Status swiglu_cuda(const tensor::Tensor &gate,
                         const tensor::Tensor &up,
                         tensor::Tensor &output,
                         void *stream = nullptr);

} // namespace kernel