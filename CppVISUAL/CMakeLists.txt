cmake_minimum_required(VERSION 3.10)
project(PC_Detector)

# 设置C++标准
set(CMAKE_CXX_STANDARD 14)

# 查找包
find_package(OpenCV REQUIRED)
find_package(cpprestsdk REQUIRED)

# 创建可执行文件
add_executable(pc_detector pc_detector.cpp)

# 链接库
target_link_libraries(pc_detector 
    ${OpenCV_LIBS} 
    cpprestsdk::cpprest
)

# 包含目录
target_include_directories(pc_detector PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)