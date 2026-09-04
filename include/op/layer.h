#pragma once

#include "base/base.h"
#include "tensor/tensor.h"
#include "base/cuda_config.h"

#include<vector>
#include<string>
#include <cstdint>

namespace op
{



using TensorInputs = std::vector<const tensor::Tensor *>; 
using TensorOutputs = std::vector<tensor::Tensor *>; 



enum class LayerType : uint8_t{
    kLayerUnknown = 0,
    kLayerLinear = 1,
    kLayerEncode = 2,
    kLayerEmbedding = 3,
    kLayerRMSNorm = 4,
    kLayerMatmul = 5,
    kLayerRope = 6 , 
    kLayerMHA = 7 , 
    kLayerSoftMax = 8,
    kLayerAdd = 9,
    kLayerSwiGLU = 10,
};

class BaseLayer
{
public:
    explicit BaseLayer(
        base::DeviceType device_type_ = base::DeviceType::kDeviceUnknown,
        LayerType layer_type_ = LayerType::kLayerUnknown,
        base::DataType data_type_ = base::DataType::kDataTypeUnknown,
        std::string layer_name_ = ""
    );

    base::DataType data_type() const;

    LayerType layer_type() const;

    virtual base::Status init() = 0;

    virtual base::Status forward() = 0;


    const std::string& get_layer_name() const;

    void set_layer_name(const std::string& layer_name) ;

    base::DeviceType device_type() const;

    void set_device_type(base::DeviceType device_type);

    virtual ~BaseLayer() = default;

private:
    std::string layer_name_; // 层名称
    LayerType layer_type_ = LayerType::kLayerUnknown; // 层类型
    base::DataType data_type_ = base::DataType::kDataTypeUnknown; // 数据类型
    base::DeviceType device_type_ = base::DeviceType::kDeviceUnknown; // 设备类型
};


class Layer 
{
    
public:
    virtual ~Layer() = default;

    virtual base::Status forward(
        const TensorInputs& inputs,
        const TensorOutputs& outputs
    ) const = 0;
};


} // namespace op
