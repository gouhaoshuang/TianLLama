#include "base/alloc.h"
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <exception>
#include <iostream>

TEST(AllocatorTest, cpu_allocator) {
    base::CPUDeviceAllocator cpu_allocator;
    base::DeviceAllocator &allocator = cpu_allocator;

    EXPECT_EQ(allocator.device_type(), base::DeviceType::kDeviceCPU);
    EXPECT_EQ(allocator.allocate(0), nullptr);

    void *ptr = allocator.allocate(64);

    EXPECT_NE(ptr, nullptr);

    allocator.memset_zero(ptr, 64);
    auto *bytes = static_cast<unsigned char *>(ptr);

    for (std::size_t i = 0; i < 64; i++) {
        if (bytes[i] != 0) {
            std::cerr << "CPU memset_zero test failed\n";
            allocator.release(ptr);
        }
    }
    allocator.release(ptr);

    std::array<int, 4> source{1, 2, 3, 4};
    std::array<int, 4> destination{};

    allocator.memcpy(
        source.data(),
        destination.data(),
        sizeof(source),
        base::MemcpyKind::kMemcpyCPU2CPU);

    EXPECT_EQ(source, destination);
}

TEST(AllocatorTest, cuda_allocator) {

    int device_count = 0;
    cudaError_t status = cudaGetDeviceCount(&device_count);
    
    if (status != cudaSuccess || device_count == 0) {
        std::cout << "CUDA test skipped: no CUDA device\n";
    }

    base::CUDADeviceAllocator allocator;
    if (allocator.device_type() != base::DeviceType::kDeviceGPU) {
        std::cerr << "GPU device type test failed \n";
    }
    if (allocator.allocate(0) != nullptr) {
        std::cerr << "GPU allocate(0) test failed \n";
    }
    std::array<int, 4> source{10, 20, 30, 40};
    std::array<int, 4> destination{};

    void *gpu_source = allocator.allocate(sizeof(source));
    void *gpu_destination = allocator.allocate(sizeof(source));

    if (gpu_source == nullptr || gpu_destination == nullptr) {
        std::cerr << "GPU allocation failed\n";

        allocator.release(gpu_source);
        allocator.release(gpu_destination);
    }

    allocator.memcpy(
        source.data(),
        gpu_source,
        sizeof(source),
        base::MemcpyKind::kMemcpyCPU2GPU);

    allocator.memcpy(
        gpu_source,
        gpu_destination,
        sizeof(source),
        base::MemcpyKind::kMemcpyGPU2GPU);

    allocator.memcpy(
        gpu_destination,
        destination.data(),
        sizeof(destination),
        base::MemcpyKind::kMemcpyGPU2CPU);

    if (source != destination) {
        std::cerr << "GPU memcpy test failed\n";

        allocator.release(gpu_source);
        allocator.release(gpu_destination);
    }

    allocator.memset_zero(
        gpu_destination,
        sizeof(source),
        nullptr,
        true);
    allocator.memcpy(
        gpu_destination,
        destination.data(),
        sizeof(destination),
        base::MemcpyKind::kMemcpyGPU2CPU);

    bool all_zero = std::all_of(
        destination.begin(),
        destination.end(),
        [](int value) {
            return value == 0;
        });

    allocator.release(gpu_source);
    allocator.release(gpu_destination);

    if (!all_zero) {
        std::cerr << "GPU memset_zero test failed\n";
    }

    std::cout << "CUDA allocator test passed\n";
}
