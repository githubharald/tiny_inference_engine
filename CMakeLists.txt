project(tiny_inference_engine)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -ffast-math -flto -fno-plt")


add_executable(tiny_inference_engine
        src/main.cpp
        src/tensor.cpp
        src/model.cpp
        src/utils.cpp
        src/conv2d.cpp
        src/core.cpp
        src/operation.cpp
        src/image.cpp
)