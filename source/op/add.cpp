#include "op/add.h"
#include "kernel/add.h"

#include <cstddef>

namespace op
{

base::Status AddLayer::forward(
    const TensorInputs &inputs,
    const TensorOutputs &outputs) const
{

    // 第一步：检查输入输出数量
    if (inputs.size() != 2)
    {
        return {
            base::kInvalidArgument,
            "AddLayer 必须要两个输入参数！！"};
    }
    if (outputs.size() != 1)
    {
        return {
            base::kInvalidArgument,
            "AddLayer 必须要一个输出参数！！"};
    }

    // 第二步：检查 Tensor 对象指针
    if (inputs[0] == nullptr ||
        inputs[1] == nullptr ||
        outputs[0] == nullptr)
    {
        return {
            base::kInvalidArgument,
            "AddLayer 接收到了一个空 Tensor"};
    }

    const tensor::Tensor &left = *inputs[0];
    const tensor::Tensor &right = *inputs[1];
    tensor::Tensor &output = *outputs[0];

    // 第三步：检查数据类型
    if (left.data_type() != base::DataType::kDataTypeFp32 ||
        right.data_type() != base::DataType::kDataTypeFp32 ||
        output.data_type() != base::DataType::kDataTypeFp32)
    {
        return {
            base::kInvalidArgument,
            "AddLayer 目前 只支持  FP32 推理"};
    }

    // 第四步：检查设备类型

    const base::DeviceType device_type = left.device_type();

    if (right.device_type() != device_type ||
        output.device_type() != device_type)
    {
        return {
            base::kInvalidArgument,
            "AddLayer 需要所有的张量都处于相同设备"};
    }

    if (device_type != base::DeviceType::kDeviceCPU &&
        device_type != base::DeviceType::kDeviceGPU)
    {
        return {
            base::kInvalidArgument,
            "AddLayer 接收到了一个不支持的设备类型"};
    }

    // 第五步：检查 shape
    if (left.dims() != right.dims() ||
        left.dims() != output.dims())
    {
        return {
            base::kInvalidArgument,
            "AddLayer Tensor shapes must be equal"};
    }

    // 第六步：检查底层内存
    if (left.ptr<float>() == nullptr ||
        right.ptr<float>() == nullptr ||
        output.ptr<float>() == nullptr)
    {
        return {
            base::kInvalidArgument,
            "AddLayer Tensor memory is empty"};
    }

    switch (device_type)
    {
    case base::DeviceType::kDeviceCPU:
        return kernel::add_cpu(left, right, output);
    case base::DeviceType::kDeviceGPU:
        return kernel::add_cuda(left, right, output);
    case base::DeviceType::kDeviceUnknown:
    default:
        return {
            base::kInvalidArgument,
            "AddLayer received an unsupported device type"};
    }
}

} // namespace op
