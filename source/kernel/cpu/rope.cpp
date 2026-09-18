#include "kernel/rope.h"
#include "op/rope.h"
#include <cmath>
#include <cstddef>

namespace kernel {

base::Status rope_cpu(const tensor::Tensor& input,
                      tensor::Tensor& output,
                      std::int64_t position,
                      double theta,
                      base::RopeLayout layout) {

    const size_t heads = static_cast<size_t>(input.dim(0));
    const size_t head_dim = static_cast<size_t>(input.dim(1));
    size_t half = head_dim / 2;

    const float* x = input.ptr<float>();
    float* y = output.ptr<float>();

    for (size_t h = 0; h < heads; h++) {
        size_t start = h * head_dim;
        for (size_t j = 0; j < half; j++) {

            // d=2*j，所以指数是 -d/head_dim
            const double freq = std::pow(theta, static_cast<double>(j) * -2.0F / head_dim);
            const double angle = static_cast<double>(position) * freq;

            const double c = std::cos(angle);
            const double s = std::sin(angle);

            const size_t d0 = layout == base::RopeLayout::kHalfSplit ? j : 2 * j;
            const size_t d1 = layout == base::RopeLayout::kHalfSplit ? j + half : 2 * j + 1;
            const size_t i0 = start + d0;
            const size_t i1 = start + d1;

            const double a = x[i0];
            const double b = x[i1];

            y[i0] = static_cast<float>(a * c - b * s);
            y[i1] = static_cast<float>(a * s + b * c);
        }
    }
    return {};
}

} // namespace kernel
