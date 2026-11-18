cmake_minimum_required(VERSION 3.28)
project(tiny_inference_engine)
set(CMAKE_CXX_STANDARD 20)

add_compile_definitions(EIGEN_USE_BLAS)
find_package(BLAS REQUIRED)

add_executable(tiny_inference_engine
        src/main.cpp
        src/tensor.cpp
        src/model.cpp
        src/utils.cpp
        src/conv2d.cpp
        src/maxpool2d.cpp
        src/core.cpp
        src/operation.cpp
        src/image.cpp
)

target_link_libraries(tiny_inference_engine PRIVATE ${BLAS_LIBRARIES})
