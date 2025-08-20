cmake_minimum_required(VERSION 3.8)
project(bird_repellent_msgs)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_compile_options(-Wall -Wextra -Wpedantic -Wno-deprecated-declarations)
add_compile_definitions(BOOST_ALLOW_DEPRECATED_HEADERS)

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

set(msg_files
  "msg/PanTiltControlMode.msg"
  "msg/Direction.msg"
  "msg/BoundingBox2d.msg"
  "msg/DetectedObject.msg"
  "msg/DetectedObjects.msg"
  "msg/PanTiltPose.msg"
  "msg/PanTiltCommand.msg"
  "msg/SpeakerCommand.msg"
  "msg/FlockAnalysis.msg"
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  DEPENDENCIES std_msgs geometry_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_export_include_directories(include)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
