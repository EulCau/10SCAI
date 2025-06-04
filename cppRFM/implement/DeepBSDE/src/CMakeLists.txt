# 自动收集所有源文件
file(GLOB_RECURSE SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
message(STATUS "Source files: ${SRC_FILES}")

# 构建可执行文件
add_executable(bsde_solver)
target_sources(bsde_solver PRIVATE ${SRC_FILES})

# 链接 libtorch
find_package(Torch REQUIRED)
target_link_libraries(bsde_solver "${TORCH_LIBRARIES}")
target_include_directories(bsde_solver PRIVATE ${TORCH_INCLUDE_DIRS})

# 设置位置无关代码 (libtorch 要求)
set_property(TARGET bsde_solver PROPERTY POSITION_INDEPENDENT_CODE ON)

# 设置输出目录
set_target_properties(bsde_solver PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin
)

# 找到 torch 的 bin 目录 (包含 DLL)
set(TORCH_DLL_DIR "$ENV{TORCH_DIR}/lib")
file(GLOB TORCH_DLLS "${TORCH_DLL_DIR}/*.dll")

# 拷贝 DLL 到输出目录 (bin)
foreach(dll ${TORCH_DLLS})
    add_custom_command(TARGET bsde_solver POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${dll}"
        $<TARGET_FILE_DIR:bsde_solver>)
endforeach()

# 定义 config 文件路径
set(CONFIG_DIR "${CMAKE_SOURCE_DIR}/configs")
file(GLOB CONFIG_FILES "${CONFIG_DIR}/*.json")

# 拷贝 config 文件到 bin 目录
foreach(cfg ${CONFIG_FILES})
    add_custom_command(TARGET bsde_solver POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${cfg}"
        $<TARGET_FILE_DIR:bsde_solver>)
endforeach()
