#pragma once

#include "op/layer.h"

#include<memory>
#include <stdexcept>
#include <utility>

namespace op
{
    
class LayerParam : public Layer
{

    
public:

    const tensor::Tensor& weight() const{
        return * weight_;
    }

protected:

    explicit LayerParam(std::shared_ptr<const tensor::Tensor> weight)
    : weight_(std::move(weight)){
        if(! weight_){
            throw std::invalid_argument("LayerParam requires a weight");
        }
    }

private:
    std::shared_ptr<const tensor::Tensor> weight_;
};


} // namespace op
