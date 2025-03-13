cmake_minimum_required(VERSION 3.10)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.8/bin/nvcc")
project(YoloP_Detector LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-12.8")
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

set(TensorRT_ROOT "/home/nyx/Tools/TensorRT-10.3.0.26" CACHE PATH "TensorRT root directory")
if(NOT EXISTS ${TensorRT_ROOT})
    message(FATAL_ERROR "TensorRT root directory not found: ${TensorRT_ROOT}")
endif()

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${TensorRT_ROOT}/include
)

set(SOURCES
    src/main.cpp
    src/yolop.cpp
)

link_directories(
    ${TensorRT_ROOT}/lib
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64
)

add_executable(${PROJECT_NAME}
    ${SOURCES}
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    ${CUDA_LIBRARIES}
    ${CUDA_cudart_LIBRARY}
    ${OpenCV_LIBRARIES}
    nvinfer
    nvonnxparser
)

target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_86,code=sm_86>
)

if(CMAKE_BUILD_TYPE MATCHES Debug)
    target_compile_definitions(${PROJECT_NAME} PRIVATE DEBUG)
endif()

message(STATUS "CUDA_LIBRARIES: ${CUDA_LIBRARIES}")
message(STATUS "OpenCV_LIBRARIES: ${OpenCV_LIBRARIES}")
message(STATUS "TensorRT_ROOT: ${TensorRT_ROOT}")