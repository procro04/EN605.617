cmake_minimum_required(VERSION 3.0)
project(assignment LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Add these lines to fix the runtime library conflict
if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
    # Or alternatively, force static linking for all:
    # set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

# Define the executable
add_executable(assignment 
    assignment/assignment.cu
)

# Include the module5/ root so CodeTimer.h is accessible
target_include_directories(assignment PRIVATE ${CMAKE_SOURCE_DIR})

# Optional: set CUDA architecture (adjust as needed, e.g. 75, 86, 89)
# Set CUDA architecture — override at configure time if needed:
#   cmake .. -DCMAKE_CUDA_ARCHITECTURES=89
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

set_target_properties(assignment PROPERTIES
    CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
)

# Optimization flags
target_compile_options(assignment PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O2>
)
