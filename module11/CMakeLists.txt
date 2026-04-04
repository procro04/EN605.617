cmake_minimum_required(VERSION 3.0)
project(module11 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Add these lines to fix the runtime library conflict
if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
    # Or alternatively, force static linking for all:
    # set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

find_package(OpenCL REQUIRED)

add_executable(OpenCLInfo
    OpenCLInfo.cpp
)
add_executable(Convolution
    Convolution.cpp
)
target_link_libraries(OpenCLInfo PRIVATE OpenCL::OpenCL)
target_link_libraries(Convolution PRIVATE OpenCL::OpenCL)
