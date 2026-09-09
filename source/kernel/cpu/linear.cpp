#include "kernel/linear.h"

#include <cmath>
#include <cstddef>

namespace kernel {

// X: [M , K] , W: [N , K] , Y: [M , N]

base::Status linear_cpu(
    const tensor::Tensor &input,
    const tensor::Tensor &weight,
    tensor::Tensor &output) {

    const size_t M = static_cast<size_t>(input.dim(0));
    const size_t K = static_cast<size_t>(input.dim(1));
    const size_t N = static_cast<size_t>(weight.dim(0));

    const float *x = input.ptr<float>();
    const float *w = weight.ptr<float>();
    float *y = output.ptr<float>();

    for (size_t m = 0; m < M; m++) {
        for (size_t n = 0; n < N; n++) {
            float sum = 0.0F;
            for (size_t k = 0; k < K; k++) {
                sum += x[m * K + k] * w[n * K + k];
            }
            y[m * N + n] = sum;
        }
    }
    return {};
}

} // namespace kernel
