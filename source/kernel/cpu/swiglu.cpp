#include "kernel/swiglu.h"

#include <cmath>
#include <cstddef>

namespace kernel

{

base::Status swiglu_cpu(
    const tensor::Tensor &gate,
    const tensor::Tensor &up,
    tensor::Tensor &output ) {

    const float *g = gate.ptr<float>();
    const float *u = up.ptr<float>();

    float *y = output.ptr<float>();

    for (size_t i = 0; i < output.size(); i++) {
        const float v = g[i];

        const float e = std::exp(-std::abs(v));

        const float sigmoid = v >= 0.0F ? 1.0F / (1.0F + e) : e / (1.0F + e);
        y[i] = (v * sigmoid) * u[i];
    }
    return {};
}

} // namespace kernel
