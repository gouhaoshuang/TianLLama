#include "base/alloc.h"
#include "op/add.h"
#include "tensor/tensor.h"


#include<cassert>
#include<iostream>
#include<memory>

int main(){

    auto allocator = std::make_shared<base::CPUDeviceAllocator>();

    tensor::Tensor left(
        {4},
        base::DataType::kDataTypeFp32,
        allocator
    );
    tensor::Tensor right(
        {4},
        base::DataType::kDataTypeFp32,
        allocator
    );
    tensor::Tensor output(
        {4},
        base::DataType::kDataTypeFp32,
        allocator
    );

    float* left_data = left.ptr<float>();
    float* right_data = right.ptr<float>();
    left_data[0] = 1.0F;
    left_data[1] = 2.0F;
    left_data[2] = 3.0F;
    left_data[3] = 4.0F;

    right_data[0] = 10.0F;
    right_data[1] = 20.0F;
    right_data[2] = 30.0F;
    right_data[3] = 40.0F;

    op::AddLayer layer;

    base::Status status = layer.forward(
        {&left, &right},
        {&output}
    );

    assert(status);

    const float* result = output.ptr<float>();

    assert(result[0] == 11.0F);
    assert(result[1] == 22.0F);
    assert(result[2] == 33.0F);
    assert(result[3] == 44.0F);

    tensor::Tensor wrong_output(
        {2},
        base::DataType::kDataTypeFp32,
        allocator
    );

    base::Status wrong_status = layer.forward(
        {&left, &right},
        {&wrong_output}
    );

    assert(!wrong_status);
    assert(wrong_status.get_err_code() == base::kInvalidArgument);


    std::cout << "AddLayer test passed\n";
    return 0;
}
