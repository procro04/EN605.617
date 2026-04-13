cmake_minimum_required(VERSION 3.10)
project(matmul_opencl LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add these lines to fix the runtime library conflict
if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
    # Or alternatively, force static linking for all:
    # set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

find_package(OpenCL REQUIRED)

add_executable(matmul matmul.cpp)
target_link_libraries(matmul PRIVATE OpenCL::OpenCL)

# Copy the kernel file next to the executable so it can be found at runtime
add_custom_command(TARGET matmul POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/matmul.cl"
        "$<TARGET_FILE_DIR:matmul>"
)
