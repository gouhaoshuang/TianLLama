#include "kernel/rope.h"
#include <cmath>
#include <cstddef>

namespace kernel {

base::Status rope_cpu(const tensor::Tensor& input,
                      tensor::Tensor& output,
                      std::int64_t position,
                      double theta) {
                        
    const size_t heads = static_cast<size_t>(input.dim(0));
    const size_t head_dim = static_cast<size_t>(input.dim(1));

    const float* x = input.ptr<float>();
    float* y = output.ptr<float>();

    for (size_t h = 0; h < heads; h++) {
        for (size_t d = 0; d < head_dim; d += 2) {
            // d=2*j，所以指数是 -d/head_dim
            const double freq = std::pow(theta, -static_cast<double>(d) / head_dim);
            const double angle = static_cast<double>(position) * freq;

            const double c = std::cos(angle);
            const double s = std::sin(angle);

            const std::size_t i = h * head_dim + d;

            const double a = x[i];
            const double b = x[i + 1];
            y[i] = static_cast<float>(a * c - b * s);
            y[i + 1] = static_cast<float>(a * s + b * c);
        }
    }
    return {};
}

} // namespace kernel
