#include "kernel/add.h"

#include <cuda_runtime.h>
#include <cstddef>
#include <string>

namespace
{
base::Status make_cuda_error(
    cudaError_t error,
    const char *operation)
{
    return {
        base::kInternalError,
        std::string(operation) + " failed: " + cudaGetErrorString(error)};
}

__global__ void add_fp32_kernel(
    const float *left,
    const float *right,
    float *output,
    std::size_t element_count)
{
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < element_count)
    {
        output[index] = left[index] + right[index];
    }
}

} // namespace

namespace kernel
{

base::Status add_cuda(
    const tensor::Tensor &left,
    const tensor::Tensor &right,
    tensor::Tensor &output
){
    const std::size_t element_count = output.size();

    if (element_count == 0)
    {
        return {};
    }

    const float *left_data = left.ptr<float>();
    const float *right_data = right.ptr<float>();
    float *output_data = output.ptr<float>();

    std::size_t threads_per_block = 256;

    std::size_t block_count = (element_count + threads_per_block - 1) / threads_per_block; // 向上取整

    add_fp32_kernel<<<block_count, threads_per_block>>>(
        left_data, right_data, output_data, element_count);

    cudaError_t status = cudaGetLastError();


    if(status != cudaSuccess){
        return make_cuda_error(
            status,
            "add_fp32_kernel launch"
        );
    }

    status = cudaDeviceSynchronize();

    if (status != cudaSuccess) {
        return make_cuda_error(
            status,
            "add_fp32_kernel execution"
        );
    }

    return {};
}


} // namespace kernel
