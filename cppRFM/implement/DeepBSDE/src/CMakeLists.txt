# 自动添加 src 目录下所有 .cpp 文件
file(GLOB SRC_FILES "*.cpp")

# 创建可执行文件
add_executable(bsde_solver ${SRC_FILES})

# 链接 libtorch
target_link_libraries(bsde_solver "${TORCH_LIBRARIES}")
target_include_directories(bsde_solver PRIVATE ${TORCH_INCLUDE_DIRS})

# 设置位置无关代码 (libtorch 要求)
set_property(TARGET bsde_solver PROPERTY POSITION_INDEPENDENT_CODE ON)

# 设置可执行文件的输出目录为 ../bin
set_target_properties(bsde_solver PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin
)
