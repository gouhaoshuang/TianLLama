cmake_minimum_required(VERSION 3.17)

project(TianLLama LANGUAGES CXX)

find_package(CUDAToolkit REQUIRED)
find_package(GTest CONFIG REQUIRED)

include(GoogleTest)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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

gtest_discover_tests(test_add_layer)

enable_testing()

add_test(
    NAME test_allocator
    COMMAND test_allocator
)
add_test(
    NAME test_buffer
    COMMAND test_buffer
)
add_test(
    NAME test_tensor
    COMMAND test_tensor
)
