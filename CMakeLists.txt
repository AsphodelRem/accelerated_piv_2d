cmake_minimum_required(VERSION 3.24)
project(piv LANGUAGES C CXX CUDA)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)

set(THREADS_PREFER_PTHREAD_FLAG ON)

find_package(Threads REQUIRED)

include_directories(.)
include_directories(image core utils)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(LIBS cufft nppc nppicc nppist X11)

file(GLOB CORE_SOURCE_FILES
        "lib/core/*.cu"
        "lib/core/*.cpp"
        "lib/image/*.cu"
        "lib/image/*.cuh"
        "lib/utils/*.cuh"
        "lib/utils/*.cu"
        "lib/core/math/*.cu"
        "lib/core/math/*.cuh"
)
message("Source files: ${CORE_SOURCE_FILES}")

add_library(piv STATIC ${CORE_SOURCE_FILES})
set_target_properties(piv PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(piv ${LIBS} ${OpenCV_LIBS})

# add_executable(piv_test_lib lib/main.cu)
# target_link_libraries(piv_test_lib piv)

include_directories(lib python-wrapper extern)
add_subdirectory(extern/pybind11)

pybind11_add_module(accelerated_piv_cpp python-wrapper/cpp_to_python.cpp ${CORE_SOURCE_FILES})
target_link_libraries(accelerated_piv_cpp PUBLIC ${LIBS} ${OpenCV_LIBS})
target_compile_definitions(accelerated_piv_cpp
        PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})