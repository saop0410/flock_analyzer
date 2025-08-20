#include "rclcpp/rclcpp.hpp"
#include "bird_detector.hpp"

int main(int argc, char * argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions option; 
  rclcpp::spin(std::make_shared<bird_detector::BirdDetectorComponent>(option));

  rclcpp::shutdown();
  return 0;
}