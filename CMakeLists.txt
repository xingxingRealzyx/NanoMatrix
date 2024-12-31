cmake_minimum_required(VERSION 3.10)
project(linalg VERSION 1.0)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 添加可执行文件
add_executable(test_linalg test_linalg.cpp)

# 设置头文件搜索路径（当前目录）
target_include_directories(test_linalg PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# 启用测试
enable_testing()
add_test(NAME test_linalg COMMAND test_linalg)

# 如果是 Debug 模式，添加调试标志
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(test_linalg PRIVATE -g -O0)
endif()

# 添加警告标志
if(MSVC)
    target_compile_options(test_linalg PRIVATE /W4 /WX)
else()
    target_compile_options(test_linalg PRIVATE 
        -Wall 
        -Wextra 
        -Wpedantic 
        -Werror 
        -Wno-unused-parameter
        -Wno-unused-variable
    )
endif() 