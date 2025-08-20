cmake_minimum_required(VERSION 3.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# Create library
project(flock_analyzer_hailo)

# Common includes
add_library (${PROJECT_NAME}
            src/flock_analyzer.cpp
            src/formation_classifier.cpp
            src/flock_analyzer_node.cpp
            include/flock_analyzer.hpp # Assuming flock_analyzer.hpp exists in include
            include/formation_classifier.hpp # Assuming formation_classifier.hpp exists in include
)
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/include)

# Link hailo_helper module
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../hailo_helper/include)
target_link_libraries(${PROJECT_NAME} hailo_helper)

# Link OpenCV module
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

# Link Torch module
# find_package(Torch REQUIRED)
# target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES})