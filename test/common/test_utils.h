#pragma once

#include "tensor/tensor.h"
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

namespace test_utils {

namespace detail {
inline void check_fp32(const tensor::Tensor& t) {
    if (t.empty() || t.data_type() != base::DataType::kDataTypeFp32 ||
        (t.device_type() != base::DeviceType::kDeviceCPU &&
         t.device_type() != base::DeviceType::kDeviceGPU)) {
        throw std::invalid_argument("Test helper requires non-empty CPU/GPU FP32 Tensor");
    }
}

} // namespace detail

// CPU vector -> 已分配的 CPU/GPU Tensor，也可以写入槽位视图。
inline void write_fp32(
    tensor::Tensor& dst,
    const std::vector<float>& datas,
    void* stream = nullptr) {
    detail::check_fp32(dst);
    if (dst.size() != datas.size()) {
        throw std::invalid_argument("Test data size mismatch");
    }

    const auto kind = dst.device_type() == base::DeviceType::kDeviceCPU
                          ? base::MemcpyKind::kMemcpyCPU2CPU
                          : base::MemcpyKind::kMemcpyCPU2GPU;
    base::CPUDeviceAllocator copier;
    copier.memcpy(datas.data(), dst.ptr<float>(), dst.byte_size(), kind, stream, true);
}
// CPU/GPU Tensor -> CPU vector，返回前复制已经完成。
inline std::vector<float> read_fp32(
    const tensor::Tensor& src,
    void* stream = nullptr) {

    detail::check_fp32(src);
    std::vector<float> values(src.size());
    const auto kind = src.device_type() == base::DeviceType::kDeviceCPU
                          ? base::MemcpyKind::kMemcpyCPU2CPU
                          : base::MemcpyKind::kMemcpyGPU2CPU;
    base::CPUDeviceAllocator copier;
    copier.memcpy(src.ptr<float>(), values.data(), src.byte_size(), kind, stream, true);
    return values;
}


inline tensor::Tensor make_cpu(
    const std::vector<std::int64_t>& shape,
    const std::vector<float>& values) {

    auto cpu = std::make_shared<base::CPUDeviceAllocator>();
    tensor::Tensor result(shape, base::DataType::kDataTypeFp32, cpu);
    write_fp32(result, values);
    return result;
}



// 已有 Tensor -> 已有 Tensor，支持四种设备复制方向。
inline void copy_fp32(
    const tensor::Tensor& src,
    tensor::Tensor& dst,
    void* stream = nullptr) {

    detail::check_fp32(src);
    detail::check_fp32(dst);

    if (!src.same_shape(dst) || src.overlaps(dst)) {
        throw std::invalid_argument("Test copy requires matching shape and separate storage");
    }

    const bool src_gpu = src.device_type() == base::DeviceType::kDeviceGPU;
    const bool dst_gpu = dst.device_type() == base::DeviceType::kDeviceGPU;

    auto kind = base::MemcpyKind::kMemcpyCPU2CPU;
    if (src_gpu) {
        kind = dst_gpu ? base::MemcpyKind::kMemcpyGPU2GPU
                       : base::MemcpyKind::kMemcpyGPU2CPU;
    } else if (dst_gpu) {
        kind = base::MemcpyKind::kMemcpyCPU2GPU;
    }
    base::CPUDeviceAllocator copier;
    copier.memcpy(src.ptr<float>(), dst.ptr<float>(), src.byte_size(), kind, stream, true);
}

// 返回独立的 GPU 副本；原 Tensor 的设备和数据不变。
inline tensor::Tensor to_gpu(const tensor::Tensor& src,
                             void* stream = nullptr) {
    detail::check_fp32(src);

    auto gpu = std::make_shared<base::CUDADeviceAllocator>();
    tensor::Tensor result(src.dims(), base::DataType::kDataTypeFp32, gpu);
    copy_fp32(src, result, stream);
    return result;
}
inline tensor::Tensor to_cpu(const tensor::Tensor& src,
                             void* stream = nullptr) {
    detail::check_fp32(src);
    auto cpu = std::make_shared<base::CPUDeviceAllocator>();
    tensor::Tensor result(src.dims(), base::DataType::kDataTypeFp32, cpu);
    copy_fp32(src, result, stream);
    return result;
}


// expected 是手算或其他参考数据；默认只使用绝对误差。
inline void expect_near(const tensor::Tensor& actual,
                        const std::vector<float>& expected,
                        double atol = 1e-5,
                        double rtol = 0.0,
                        void* stream = nullptr) {
    const auto values = read_fp32(actual, stream);
    ASSERT_EQ(values.size(), expected.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        ASSERT_TRUE(std::isfinite(values[i])) << "index=" << i;
        ASSERT_TRUE(std::isfinite(expected[i])) << "index=" << i;
        EXPECT_LE(std::abs(double(values[i]) - double(expected[i])),
                  atol + rtol * std::abs(double(expected[i])))
            << "index=" << i << " actual=" << values[i] << " expected=" << expected[i];
    }
}

// expected 也可以是 CPU/GPU Tensor；还要检查 shape 一致。
inline void expect_near(const tensor::Tensor& actual,
                        const tensor::Tensor& expected,
                        double atol = 1e-5,
                        double rtol = 0.0,
                        void* stream = nullptr) {
    ASSERT_EQ(actual.dims(), expected.dims());
    expect_near(actual, read_fp32(expected, stream), atol, rtol, stream);
}
} // namespace test_utils
