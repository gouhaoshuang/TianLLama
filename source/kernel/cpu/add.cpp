#include "kernel/add.h"

#include <cstddef>


namespace kernel{


base::Status add_cpu(
    const tensor::Tensor& left,
    const tensor::Tensor& right,
    tensor::Tensor&  output
){

    const float* left_data = left.ptr<float>();
    const float* right_data = right.ptr<float>();
    float* output_data = output.ptr<float>();

    for(std::size_t i = 0 ; i < output.size() ;i ++){
        output_data[i] = left_data[i] + right_data[i];
    }

    return {};

}

}