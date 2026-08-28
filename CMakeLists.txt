cmake_minimum_required(VERSION 3.16)

project(TianLLama LANGUAGES CXX)

add_library(tianllama
    source/base/alloc.cpp
)

target_include_directories(tianllama
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(tianllama
    PUBLIC
        cxx_std_20
)

add_executable(test_allocator
    test/optimized/test_allocator.cpp
)

target_link_libraries(test_allocator
    PRIVATE
        tianllama
)

enable_testing()

add_test(
    NAME test_allocator
    COMMAND test_allocator
)
