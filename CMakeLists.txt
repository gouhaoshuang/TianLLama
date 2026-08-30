cmake_minimum_required(VERSION 3.17)

project(TianLLama LANGUAGES CXX)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(tianllama
    source/base/alloc.cpp
    source/base/alloc_cpu.cpp
    source/base/alloc_cu.cpp
    source/base/buffer.cpp
)

target_include_directories(tianllama
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(tianllama
    PUBLIC
        cxx_std_20
)


target_link_libraries(tianllama
    PUBLIC
        CUDA::cudart
)


add_executable(test_allocator
    test/optimized/test_allocator.cpp
)
target_link_libraries(test_allocator
    PRIVATE
        tianllama
)
add_executable(test_buffer
    test/optimized/test_buffer.cpp
)

target_link_libraries(test_buffer
    PRIVATE
        tianllama
)


enable_testing()

add_test(
    NAME test_allocator
    COMMAND test_allocator
)
add_test(
    NAME test_buffer
    COMMAND test_buffer
)