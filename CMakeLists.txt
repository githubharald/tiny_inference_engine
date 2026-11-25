cmake_minimum_required(VERSION 3.28)
project(tiny_inference_engine LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)


# the core as a lib
find_package(BLAS REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
add_library(tiny_inference_engine_core OBJECT
        src/tensor.cpp
        src/model.cpp
        src/utils.cpp
        src/conv2d.cpp
        src/maxpool2d.cpp
        src/core.cpp
        src/operation.cpp
        src/image.cpp
)
target_include_directories(tiny_inference_engine_core PUBLIC src)
target_compile_definitions(tiny_inference_engine_core PUBLIC EIGEN_USE_BLAS)
target_link_libraries(tiny_inference_engine_core PUBLIC BLAS::BLAS Eigen3::Eigen)


# the executable (the demo)
add_executable(tiny_inference_engine src/main.cpp)
target_link_libraries(tiny_inference_engine PRIVATE tiny_inference_engine_core)


# and the tests
include(CTest)
if (BUILD_TESTING)
    # list of the tests
    set(TEST_TARGETS
            test_conv2d
            test_maxpool2d
            test_core
            test_utils
    )
    # one executable per test
    foreach (test_name ${TEST_TARGETS})
        add_executable(${test_name} tests/${test_name}.cpp)
        target_link_libraries(${test_name} PRIVATE tiny_inference_engine_core)
        add_test(NAME ${test_name} COMMAND ${test_name})
    endforeach ()
endif ()