cmake_minimum_required(VERSION 3.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# Create library
project(detection_processor)

# Common includes
add_library (${PROJECT_NAME}
            src/detection_processor.cpp
            include/detection_processor.hpp
)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)

# Link object_detector module
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../object_detector/include)
target_link_libraries(${PROJECT_NAME} object_detector)

# Link hailo_helper module
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../hailo_helper hailo_helper)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../hailo_helper/include)
target_link_libraries(${PROJECT_NAME} hailo_helper)

# Link object_detector module
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../object_detector object_detector)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../object_detector/include)
target_link_libraries(${PROJECT_NAME} object_detector)

# Link OpenCV module
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})