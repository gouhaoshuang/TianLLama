#include "kernel/softmax.h"

#include <algorithm>
#include <cmath>

#include <string>

namespace kernel {

base::Status softmax_cpu(
    const tensor::Tensor& input,
    tensor::Tensor& output) {

    const auto rows = static_cast<size_t>(input.dim(0));
    const auto cols = static_cast<size_t>(input.dim(1));

    const float* x = input.ptr<float>();
    float* y = output.ptr<float>();

    for (std::size_t r = 0; r < rows; r++) {
        size_t offset = r * cols;

        float maximum = x[offset];
        for (size_t j = 1; j < cols; j++) {
            maximum = std::max(maximum, x[offset + j]);
        }

        float sum = 0.0F;

        for (size_t j = 0; j < cols; j++) {
            sum += std::exp(x[offset + j] - maximum);
        }
        for (size_t j = 0; j < cols; j++) {
            y[offset + j] = std::exp(x[offset + j] - maximum) / sum;
        }
    }
    return {};
}

} // namespace kernel
