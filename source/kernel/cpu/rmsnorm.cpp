#include "kernel/rmsnorm.h"

#include <cmath>
#include <cstddef>

namespace kernel {

base::Status rmsnorm_cpu(
    const tensor::Tensor &input,
    const tensor::Tensor &weight,
    tensor::Tensor &output,
    float epsilon) {

    const size_t dim = weight.size();
    const std::size_t rows = input.size() / dim;

    const float *x = input.ptr<float>();
    const float *w = weight.ptr<float>();
    float *y = output.ptr<float>();

    for (size_t row = 0; row < rows; row++) {
        const size_t offset = row * dim;
        float sum = 0.0F;

        for (size_t j = 0; j < dim; j++) {
            const float value = x[offset + j];
            sum += value * value;
        }

        const float scale = 1.0F / std::sqrt(sum / dim + epsilon);

        for (size_t j = 0; j < dim; j++) {
            y[offset + j] = x[offset + j] * scale * w[j];
        }
    }
    return {};
}

} // namespace kernel
