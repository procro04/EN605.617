cmake_minimum_required(VERSION 4.0)
project(HelloWorld LANGUAGES CXX CUDA C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Add these lines to fix the runtime library conflict
if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
    # Or alternatively, force static linking for all:
    # set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

find_package(OpenCL REQUIRED)

add_executable(helloworld-cpp
    helloworld.cpp
)
add_executable(helloworld-cu
    hello-world.cu
)
add_executable(helloworld-cl
    hello_world_cl.c
)
add_executable(cl-device-query
    clDeviceQuery.cpp
)
target_link_libraries(helloworld-cl PRIVATE OpenCL::OpenCL)
target_link_libraries(cl-device-query PRIVATE OpenCL::OpenCL)

set_target_properties(helloworld-cu PROPERTIES
    CUDA_ARCHITECTURES native
)
