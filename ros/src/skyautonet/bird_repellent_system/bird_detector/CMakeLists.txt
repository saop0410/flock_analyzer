cmake_minimum_required(VERSION 3.8)
project(bird_detector)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_compile_options(-Wall -Wextra -Wpedantic -Wno-deprecated-declarations)
add_compile_definitions(BOOST_ALLOW_DEPRECATED_HEADERS)

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

# Create library
ament_auto_add_library(${PROJECT_NAME} SHARED
  nodes/bird_detector_node.cpp
  src/bird_detector.cpp
)

# Common includes
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/include)

# Link detection_processor module
add_subdirectory(libs/detection_processor detection_processor)
target_include_directories(${PROJECT_NAME} PRIVATE libs/detection_processor/include)
target_link_libraries(${PROJECT_NAME} detection_processor)

# Link OpenCV module
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

rclcpp_components_register_node(${PROJECT_NAME}
  PLUGIN "bird_detector::BirdDetectorComponent"
  EXECUTABLE ${PROJECT_NAME}_node
)

install(TARGETS
  ${PROJECT_NAME}
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_auto_package(
  INSTALL_TO_SHARE
  config
  models
  launch
)