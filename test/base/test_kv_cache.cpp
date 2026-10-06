#include "base/kv_cache.h"
#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {

// 本文件只写入 [1, 2] 槽位，每次 data 必须恰好包含两个元素。
void write_data(tensor::Tensor& slot, const std::vector<float>& data) {
    base::CPUDeviceAllocator copier;
    const auto kind = slot.device_type() == base::DeviceType::kDeviceGPU
                          ? base::MemcpyKind::kMemcpyCPU2GPU
                          : base::MemcpyKind::kMemcpyCPU2CPU;

    copier.memcpy(data.data(), slot.ptr<float>(), slot.byte_size(), kind, nullptr, true);
}

std::vector<float> read_data(const tensor::Tensor& tensor) {
    std::vector<float> data(tensor.size());
    base::CPUDeviceAllocator copier;
    const auto kind = tensor.device_type() == base::DeviceType::kDeviceGPU
                          ? base::MemcpyKind::kMemcpyGPU2CPU
                          : base::MemcpyKind::kMemcpyCPU2CPU;
    copier.memcpy(tensor.ptr<float>(), data.data(), tensor.byte_size(), kind, nullptr, true);
    return data;
}

void check_lifecycle(base::KVCache& cache, base::DeviceType device) {

    // 1. 初始为空；不能读前缀，也不能跳过位置 0。
    ASSERT_EQ(cache.device_type(), device);
    ASSERT_EQ(cache.capacity(), 2);
    ASSERT_EQ(cache.length(), 0);
    EXPECT_THROW(cache.keys(), std::logic_error);
    EXPECT_THROW(cache.values(), std::logic_error);
    EXPECT_THROW(cache.key_slot(1), std::out_of_range);
    EXPECT_EQ(cache.commit(1).get_err_code(), base::kInvalidArgument);
    EXPECT_EQ(cache.length(), 0);

    // 2. 写入第一个 token；写槽位不会自动增加长度。
    tensor::Tensor k0 = cache.key_slot(0);
    tensor::Tensor v0 = cache.value_slot(0);
    ASSERT_EQ(k0.device_type(), device);
    ASSERT_EQ(v0.device_type(), device);

    const float* storage_start = k0.ptr<float>();

    write_data(k0, {1, 2});
    write_data(v0, {3, 4});
    EXPECT_EQ(cache.length(), 0);
    ASSERT_TRUE(cache.commit(0));
    ASSERT_EQ(cache.length(), 1);

    auto keys = cache.keys();
    auto values = cache.values();
    ASSERT_EQ(keys.device_type(), device);
    ASSERT_EQ(values.device_type(), device);
    EXPECT_EQ(keys.dims(), (std::vector<std::int64_t>{1, 1, 2}));
    EXPECT_EQ(values.dims(), keys.dims());
    EXPECT_EQ(read_data(keys), (std::vector<float>{1, 2}));
    EXPECT_EQ(read_data(values), (std::vector<float>{3, 4}));
    EXPECT_THROW(cache.key_slot(0), std::out_of_range);
    EXPECT_EQ(cache.commit(0).get_err_code(), base::kInvalidArgument);
    EXPECT_EQ(cache.length(), 1);
    // 3. 写入第二个 token，完整前缀必须保留第一个 token。
    auto k1 = cache.key_slot(1);
    auto v1 = cache.value_slot(1);
    write_data(k1, {5, 6});
    write_data(v1, {7, 8});
    ASSERT_TRUE(cache.commit(1));
    ASSERT_EQ(cache.length(), 2);
    EXPECT_EQ(read_data(cache.keys()), (std::vector<float>{1, 2, 5, 6}));
    EXPECT_EQ(read_data(cache.values()), (std::vector<float>{3, 4, 7, 8}));


    // 4. 满容量后拒绝继续写入或提交。
    EXPECT_THROW(cache.key_slot(2), std::out_of_range);
    EXPECT_EQ(cache.commit(2).get_err_code(), base::kInvalidArgument);
    EXPECT_EQ(cache.length(), 2);

    // 5. reset 复用原存储，新请求只暴露新提交的一个 token。
    cache.reset();
    ASSERT_EQ(cache.length(), 0);
    EXPECT_THROW(cache.keys(), std::logic_error);
    EXPECT_THROW(cache.values(), std::logic_error);
    auto new_k = cache.key_slot(0);
    auto new_v = cache.value_slot(0);
    EXPECT_EQ(new_k.ptr<float>(), storage_start); // 只比较地址。
    write_data(new_k, {0, 1});
    write_data(new_v, {30, 40});
    ASSERT_TRUE(cache.commit(0));
    ASSERT_EQ(cache.length(), 1);
    EXPECT_EQ(cache.keys().dims(), (std::vector<std::int64_t>{1, 1, 2}));
    EXPECT_EQ(cache.values().dims(), cache.keys().dims());
    EXPECT_EQ(read_data(cache.keys()), (std::vector<float>{0, 1}));
    EXPECT_EQ(read_data(cache.values()), (std::vector<float>{30, 40}));
}

} // namespace

TEST(KVCacheTest, CpuLifecycle) {
    base::KVCache cache(2, 1, 2); // 验证旧的三参数调用默认使用 CPU。

    ASSERT_NO_FATAL_FAILURE(
        check_lifecycle(cache, base::DeviceType::kDeviceCPU));
}

TEST(KVCacheTest, GpuLifecycle) {
    auto gpu = std::make_shared<base::CUDADeviceAllocator>();
    base::KVCache cache(2, 1, 2, gpu);
    ASSERT_NO_FATAL_FAILURE(
        check_lifecycle(cache, base::DeviceType::kDeviceGPU));
}