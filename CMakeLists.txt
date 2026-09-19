cmake_minimum_required(VERSION 3.18)

project(TianLLama LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(GTest CONFIG REQUIRED)

include(GoogleTest)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89)
endif()


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


add_executable(test_allocator test/optimized/test_allocator.cpp)
target_link_libraries(test_allocator PRIVATE tianllama GTest::gtest_main)
add_executable(test_add_layer test/op/test_add_layer.cpp)
target_link_libraries(test_add_layer PRIVATE tianllama GTest::gtest_main)
add_executable(test_rmsnorm_layer test/op/test_rmsnorm_layer.cpp)
target_link_libraries(test_rmsnorm_layer PRIVATE tianllama GTest::gtest_main)
add_executable(test_linear_layer test/op/test_linear_layer.cpp)
target_link_libraries(test_linear_layer PRIVATE tianllama GTest::gtest_main)
add_executable(test_tensor test/tensor/test_tensor.cpp)
target_link_libraries(test_tensor PRIVATE tianllama GTest::gtest_main)
add_executable(test_buffer test/optimized/test_buffer.cpp)
target_link_libraries(test_buffer PRIVATE tianllama GTest::gtest_main)
add_executable(test_ffn_layer test/op/test_ffn.cpp)
target_link_libraries(test_ffn_layer PRIVATE tianllama GTest::gtest_main)
add_executable(test_rope_softmax test/op/test_rope_softmax.cpp)
target_link_libraries(test_rope_softmax PRIVATE tianllama GTest::gtest_main)
add_executable(test_attention test/op/test_attention.cpp)
target_link_libraries(test_attention PRIVATE tianllama GTest::gtest_main)
add_executable(test_decoder_block test/model/test_decoder_block.cpp)
target_link_libraries(test_decoder_block PRIVATE tianllama GTest::gtest_main)
add_executable(test_qwen2_layer0 test/model/test_qwen2_layer0.cpp)
target_link_libraries(test_qwen2_layer0 PRIVATE tianllama GTest::gtest_main)
target_compile_definitions(test_qwen2_layer0 PRIVATE
    QWEN_LAYER0_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test_data/qwen2_5_0_5b_layer0_v1"
)

gtest_discover_tests(test_buffer)
gtest_discover_tests(test_tensor)
gtest_discover_tests(test_allocator)
gtest_discover_tests(test_add_layer)
gtest_discover_tests(test_rmsnorm_layer)
gtest_discover_tests(test_linear_layer)
gtest_discover_tests(test_ffn_layer)
gtest_discover_tests(test_rope_softmax)
gtest_discover_tests(test_attention)
gtest_discover_tests(test_decoder_block)
gtest_discover_tests(test_qwen2_layer0)

enable_testing()
