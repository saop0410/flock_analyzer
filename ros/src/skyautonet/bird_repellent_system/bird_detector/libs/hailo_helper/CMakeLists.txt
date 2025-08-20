cmake_minimum_required(VERSION 3.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# Create library
project(hailo_helper)

# Common includes
add_library (${PROJECT_NAME}
            src/hailo_inference_engine.cpp
            include/hailo_helper.hpp
            include/hailo_infos.hpp
)
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/include)

# Link HailoRT module
find_package(HailoRT REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE HailoRT::libhailort)