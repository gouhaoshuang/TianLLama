#pragma once

#include "base/kv_cache.h"
#include "op/add.h"
#include "op/attention.h"
#include "op/linear.h"
#include "op/rmsnorm.h"
#include "op/swiglu.h"
#include <cstdint>
#include <memory>

namespace model {

struct DecoderConfig {
    int64_t dim = 8;
    int64_t hidden_dim = 12;
    int64_t heads = 2;
    int64_t capacity = 4;
    float epsilon = 1e-5F;
    double rope_theta = 10000.0;
};

struct DecoderWeights {
    std::shared_ptr<const tensor::Tensor> attention_norm, ffn_norm;
    std::shared_ptr<const tensor::Tensor> wq, wk, wv, wo;
    std::shared_ptr<const tensor::Tensor> gate, up, down;
};

class DecoderBlock {

  public:
    DecoderBlock(DecoderConfig config, const DecoderWeights& weights);

    base::Status forward(const tensor::Tensor& x, tensor::Tensor& y);
    int64_t length() const { return cache_.length(); }
    bool failed() const { return failed_; }
    void reset() {
        cache_.reset();
        failed_ = false;
    }

  private:
    // 声明顺序就是初始化顺序：先检查配置，再创建算子和工作区。
    DecoderConfig config_;
    op::RmsNormLayer attention_norm_, ffn_norm_;
    op::LinearLayer q_proj_, k_proj_, v_proj_, o_proj_;
    op::LinearLayer gate_proj_, up_proj_, down_proj_;

    op::AttentionLayer attention_;
    op::SwiGLULayer swiglu_;
    op::AddLayer add_;
    std::shared_ptr<base::CPUDeviceAllocator> cpu_allocator_;
    base::KVCache cache_;

    tensor::Tensor n_, q_, k_, q_rot_, a_, attn_out_, h_;
    tensor::Tensor z_, gate_, up_, act_, down_;
    bool failed_ = false;
};

} // namespace model
