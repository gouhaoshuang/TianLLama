#include "op/add.h"


#include <cstddef>

namespace op
{

base::Status AddLayer::forward(
    const TensorInputs& inputs,
    const TensorOutputs& outputs
) const {
    if(inputs.size() != 2){
        return {
            base::kInvalidArgument,
            "AddLayer 必须要两个输入参数！！"
        };
    }
    if(outputs.size() != 1){
        return {
            base::kInvalidArgument,
            "AddLayer 必须要一个输出参数！！"
        };
    }
    if (inputs[0] == nullptr ||
        inputs[1] == nullptr ||
        outputs[0] == nullptr) {
        return {
            base::kInvalidArgument,
            "AddLayer 接收到了一个空 Tensor"
        };
    }

    const tensor::Tensor& left = *inputs[0];
    const tensor::Tensor& right = *inputs[1];
    tensor::Tensor& output = *outputs[0];


    if (left.data_type() != base::DataType::kDataTypeFp32 ||
        right.data_type() != base::DataType::kDataTypeFp32 ||
        output.data_type() != base::DataType::kDataTypeFp32) {
        return {
            base::kInvalidArgument,
            "AddLayer 目前 只支持  FP32 推理"
        };
    }

    if (left.device_type() != base::DeviceType::kDeviceCPU ||
        right.device_type() != base::DeviceType::kDeviceCPU ||
        output.device_type() != base::DeviceType::kDeviceCPU) {
        return {
            base::kInvalidArgument,
            "AddLayer 目前  只支持  CPU 推理"
        };
    }
    if (left.dims() != right.dims() ||
        left.dims() != output.dims()) {
        return {
            base::kInvalidArgument,
            "AddLayer Tensor shapes must be equal"
        };
    }

    const float* left_data = left.ptr<float>();
    const float* right_data = right.ptr<float>();

    float* output_data = output.ptr<float>();

    if (left_data == nullptr ||
        right_data == nullptr ||
        output_data == nullptr) {
        return {
            base::kInvalidArgument,
            "AddLayer Tensor memory is empty"
        };
    }

    for(std::size_t i = 0 ; i < output.size() ;i ++){
        output_data[i] = left_data[i] + right_data[i];
    }

    return {}; // status 默认构造
}


} // namespace op
