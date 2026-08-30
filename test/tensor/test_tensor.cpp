#include "base/buffer.h"
#include "tensor/tensor.h"
#include <cuda_runtime_api.h>

#include<memory>
#include<cassert>
#include <iostream>


int main(){
    auto allocator =  std::make_shared<base::CPUDeviceAllocator>();

    tensor::Tensor value(
        {2, 3},
        base::DataType::kDataTypeFp32,
        allocator
    );

    assert(value.size() == 6);
    assert(value.byte_size() == 6 * sizeof(float));
    assert(value.device_type() == base::DeviceType::kDeviceCPU);

    assert(value.dims().size() == 2);
    assert(value.dims()[0] == 2);
    assert(value.dims()[1] == 3);


    float* data = value.ptr<float>();
    assert(data != nullptr);

    data[0] = 1.25F;

    const tensor::Tensor& read_only_value = value;
    const float* read_only_data =  read_only_value.ptr<float>();

    assert(read_only_data[0] == 1.25F);

    std::cout << "Tensor test passed\n";
    return 0;


}