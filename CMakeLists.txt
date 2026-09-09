cmake_minimum_required(VERSION 3.18)

project(TianLLama LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(GTest CONFIG REQUIRED)

include(GoogleTest)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89)
endif()



# add_library(tianllama
#     source/base/base.cpp
#     source/base/alloc.cpp
#     source/base/alloc_cpu.cpp
#     source/base/alloc_cu.cpp
#     source/base/buffer.cpp
#     source/tensor/tensor.cpp
#     source/op/layer.cpp
# )

file(
    GLOB_RECURSE TIANLLAMA_SOURCES
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/source/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/source/*.cu"
)

add_library(tianllama
    ${TIANLLAMA_SOURCES}
)


target_include_directories(tianllama
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(tianllama
    PUBLIC
        cxx_std_20
)
set_target_properties(tianllama PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
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
        GTest::gtest
        GTest::gtest_main
)
add_executable(test_buffer
    test/optimized/test_buffer.cpp
)

target_link_libraries(test_buffer
    PRIVATE
        tianllama
)
add_executable(test_tensor
    test/tensor/test_tensor.cpp
)

target_link_libraries(test_tensor
    PRIVATE
        tianllama
)
add_executable(test_add_layer
    test/op/test_add_layer.cpp
)

target_link_libraries(test_add_layer
    PRIVATE
        tianllama
        GTest::gtest
        GTest::gtest_main
)
add_executable(test_rmsnorm_layer
    test/op/test_rmsnorm_layer.cpp
)

target_link_libraries(test_rmsnorm_layer
    PRIVATE
        tianllama
        GTest::gtest
        GTest::gtest_main
)
add_executable(test_linear_layer
    test/op/test_linear_layer.cpp
)

target_link_libraries(test_linear_layer
    PRIVATE
        tianllama
        GTest::gtest
        GTest::gtest_main
)
add_executable(test_ffn_layer test/op/test_ffn.cpp)
target_link_libraries(test_ffn_layer PRIVATE tianllama GTest::gtest_main)

gtest_discover_tests(test_allocator)
gtest_discover_tests(test_add_layer)
gtest_discover_tests(test_rmsnorm_layer)
gtest_discover_tests(test_linear_layer)
gtest_discover_tests(test_ffn_layer)

enable_testing()


add_test(
    NAME test_buffer
    COMMAND test_buffer
)
add_test(
    NAME test_tensor
    COMMAND test_tensor
)
