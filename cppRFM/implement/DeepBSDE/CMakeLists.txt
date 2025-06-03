cmake_minimum_required(VERSION 3.10)
project(BSDE_Libtorch)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找 CUDA
find_package(CUDA REQUIRED)

if (CUDA_FOUND)
    add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
    set_target_properties(CUDA::nvToolsExt PROPERTIES
        INTERFACE_LINK_LIBRARIES "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/nvToolsExt.lib"
        INTERFACE_INCLUDE_DIRECTORIES "${CUDA_TOOLKIT_ROOT_DIR}/include"
    )
endif()

# 需设置环境变量 TORCH_DIR
if(NOT DEFINED ENV{TORCH_DIR})
    message(FATAL_ERROR "Please set TORCH_DIR environment variable to the libtorch directory.")
endif()
set(Torch_DIR "$ENV{TORCH_DIR}/share/cmake/Torch")

# 引入 libtorch
find_package(Torch REQUIRED)

# 包含子目录
add_subdirectory(src)
