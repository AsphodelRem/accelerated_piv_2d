cmake_minimum_required(VERSION 3.24)
project(piv CUDA CXX)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)

set(THREADS_PREFER_PTHREAD_FLAG ON)

find_package(Threads REQUIRED)

include_directories(piv2d)
include_directories(piv2d/image piv2d/core piv2d/utils)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(LIBS cufft nppc nppicc nppist)

file(GLOB CORE_SOURCE_FILES
        "piv2d/core/*.cpp"
        "piv2d/core/*.cu"
        "piv2d/image/*.cu"
        "piv2d/image/*.cuh"
        "piv2d/utils/*.cuh"
        "piv2d/utils/*.cu"
        "piv2d/core/math/*.cu"
        "piv2d/core/math/*.cuh"
        )

message("Source files: ${CORE_SOURCE_FILES}")

add_library(piv ${CORE_SOURCE_FILES})

set_target_properties(piv PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(piv ${LIBS})

set(OPENCV opencv_world480)

add_executable(piv_test_lib piv2d/main.cu)
target_link_libraries(piv_test_lib piv)
target_link_libraries(piv_test_lib ${OpenCV_LIBS})
