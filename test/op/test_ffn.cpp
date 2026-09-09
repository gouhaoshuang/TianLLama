#include "op/add.h"
#include "op/linear.h"
#include "op/rmsnorm.h"
#include "op/swiglu.h"

#include <cstdint>
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// 输入数据在 CPU；根据 allocator 上传到指定设备。
std::shared_ptr<tensor::Tensor> make_fp32(
    const std::vector<int64_t> &shape,
    const std::vector<float> &values,

    std::shared_ptr<base::DeviceAllocator> allocator) {

    auto t = std::make_shared<tensor::Tensor>(
        shape,
        base::DataType::kDataTypeFp32,
        allocator);
    if (values.size() != t->size()) {
        throw std::invalid_argument("Test data size mismatch");
    }
    const auto kind = t->device_type() == base::DeviceType::kDeviceCPU
                          ? base::MemcpyKind::kMemcpyCPU2CPU
                          : base::MemcpyKind::kMemcpyCPU2GPU;
    allocator->memcpy(values.data(), t->ptr<float>(), t->byte_size(), kind);
    return t;
}

// 输出统一复制到 CPU，GPU 指针不能由 CPU 直接解引用。
std::vector<float> read_fp32(
    const tensor::Tensor &t,
    const std::shared_ptr<base::DeviceAllocator> &allocator) {

    std::vector<float> values(t.size());

    const auto kind = t.device_type() == base::DeviceType::kDeviceCPU
                          ? base::MemcpyKind::kMemcpyCPU2CPU
                          : base::MemcpyKind::kMemcpyGPU2CPU;
    allocator->memcpy(t.ptr<float>(), values.data(), t.byte_size(), kind);
    return values;
}

void expect_values(const std::vector<float> &actual,
                   const std::vector<float> &expected) {
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(actual[i], expected[i], 1e-5F) << "index=" << i;
    }
}

// 仅测试辅助：非 void 函数中不用 ASSERT_* 提前返回。
void require_ok(const base::Status &status, const char *stage) {
    if (!status) {
        throw std::runtime_error(std::string(stage) + ": " + status.get_err_message());
    }
}

std::vector<float> run_fixed_ffn(
    const std::shared_ptr<base::DeviceAllocator> &allocator) {

    auto x = make_fp32({2, 2}, {1, -1, -1, 1}, allocator);
    auto gamma = make_fp32({2}, {1, 2}, allocator);
    auto wg = make_fp32({3, 2}, {1, 0, 0, 1, 1, 1}, allocator);
    auto wu = make_fp32({3, 2}, {1, 1, 1, -1, 1, 2}, allocator);
    auto wd = make_fp32({2, 3}, {1, 2, -1, -1, 1, 1}, allocator);

    const auto fp32 = base::DataType::kDataTypeFp32;
    tensor::Tensor z({2, 2}, fp32, allocator);
    tensor::Tensor gate({2, 3}, fp32, allocator);
    tensor::Tensor up({2, 3}, fp32, allocator);
    tensor::Tensor act({2, 3}, fp32, allocator);
    tensor::Tensor down({2, 2}, fp32, allocator);
    tensor::Tensor y({2, 2}, fp32, allocator);

    op::RmsNormLayer norm(gamma, 1e-5F);
    op::LinearLayer gate_proj(wg);
    op::LinearLayer up_proj(wu);
    op::SwiGLULayer swiglu;
    op::LinearLayer down_proj(wd);
    op::AddLayer add;

    // 整条链：不搬运中间数据，不覆盖 x，不重写已有算子。
    require_ok(norm.forward({x.get()}, {&z}), "RMSNorm");
    require_ok(gate_proj.forward({&z}, {&gate}), "gate Linear");
    require_ok(up_proj.forward({&z}, {&up}), "up Linear");
    require_ok(swiglu.forward({&gate, &up}, {&act}), "SwiGLU");
    require_ok(down_proj.forward({&act}, {&down}), "down Linear");
    require_ok(add.forward({x.get(), &down}, {&y}), "residual Add");

    // 计算结束后再检查各阶段，失败时可以定位到具体节点。
    {
        SCOPED_TRACE("RMSNorm");
        expect_values(read_fp32(z, allocator),
                      {0.999995F, -1.999990F, -0.999995F, 1.999990F});
    }
    {
        SCOPED_TRACE("gate/up Linear");
        expect_values(read_fp32(gate, allocator),
                      {0.999995F, -1.999990F, -0.999995F, -0.999995F, 1.999990F, 0.999995F});
        expect_values(read_fp32(up, allocator),
                      {-0.999995F, 2.999985F, -2.999985F, 0.999995F, -2.999985F, 2.999985F});
    }
    {
        SCOPED_TRACE("SwiGLU");
        expect_values(read_fp32(act, allocator),
                      {-0.731050285F, -0.715216680F, 0.806819145F, -0.268939715F, -5.284723321F, 2.193150855F});
    }
    {
        SCOPED_TRACE("down Linear");
        expect_values(read_fp32(down, allocator),
                      {-2.968302789F, 0.822652751F, -13.031537212F, -2.822632751F});
    }
    expect_values(read_fp32(*x, allocator), {1, -1, -1, 1});
    auto result = read_fp32(y, allocator);
    {
        SCOPED_TRACE("residual Add");
        expect_values(result,
                      {-1.968302789F, -0.177347249F, -14.031537212F, -1.822632751F});
    }
    return result;
}

} // namespace

TEST(SwiGLUTest, CpuFixedValues) {
    auto cpu = std::make_shared<base::CPUDeviceAllocator>();

    auto gate = make_fp32({1, 3}, {-1, 0, 1}, cpu);
    auto up = make_fp32({1, 3}, {2, 3, 4}, cpu);

    tensor::Tensor output({1, 3}, base::DataType::kDataTypeFp32, cpu);

    op::SwiGLULayer layer;

    const auto status = layer.forward({gate.get(), up.get()}, {&output});

    ASSERT_TRUE(status) << status.get_err_message();

    expect_values(read_fp32(output, cpu), {-0.537882843F, 0.0F, 2.924234315F});
    expect_values(read_fp32(*gate, cpu), {-1, 0, 1}); // 输入不能被修改。
    expect_values(read_fp32(*up, cpu), {2, 3, 4});

    // 顺便验证最基本的错误路径，不增加测试框架。
    tensor::Tensor wrong_shape({3, 1}, base::DataType::kDataTypeFp32, cpu);

    EXPECT_EQ(layer.forward({gate.get(), up.get()}, {&wrong_shape}).get_err_code(), base::kInvalidArgument);

    EXPECT_EQ(layer.forward({gate.get(), up.get()}, {gate.get()}).get_err_code(), base::kInvalidArgument);
    EXPECT_EQ(layer.forward({nullptr, up.get()}, {&output}).get_err_code(), base::kInvalidArgument);
}

TEST(FFNTest, CpuFixedChain) {
    auto cpu = std::make_shared<base::CPUDeviceAllocator>();
    run_fixed_ffn(cpu);
}

TEST(FFNTest, CudaMatchesCpu) {
    int device_count = 0;
    const auto status = cudaGetDeviceCount(&device_count);
    if (status == cudaErrorNoDevice ||
        (status == cudaSuccess && device_count == 0)) {
        GTEST_SKIP() << "No CUDA device";
    }

    ASSERT_EQ(status, cudaSuccess) << cudaGetErrorString(status);

    auto cpu = std::make_shared<base::CPUDeviceAllocator>();
    auto gpu = std::make_shared<base::CUDADeviceAllocator>();

    const auto cpu_result = run_fixed_ffn(cpu);
    const auto gpu_result = run_fixed_ffn(gpu);

    expect_values(gpu_result, cpu_result);
}
