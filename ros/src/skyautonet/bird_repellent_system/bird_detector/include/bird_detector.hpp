#ifndef BIRD_DETECTOR_COMPONENT_HPP_
#define BIRD_DETECTOR_COMPONENT_HPP_

/*** Include ***/
/* for general */
#include <string>
#include <vector>

/* for OpenCV */
#include <opencv2/opencv.hpp>

/* for ROS2 */
#include <rclcpp/rclcpp.hpp>
#include <image_transport/image_transport.hpp>

#if __has_include(<cv_bridge/cv_bridge.hpp>)
#include <cv_bridge/cv_bridge.hpp>
#else
#include <cv_bridge/cv_bridge.h>
#endif

/* for bird system msgs */
#include "bird_repellent_msgs/msg/detected_object.hpp"
#include "bird_repellent_msgs/msg/detected_objects.hpp"

/* for My modules */
#include "detection_processor.hpp"

namespace bird_detector {

using DetectedObjectMsg = bird_repellent_msgs::msg::DetectedObject;
using DetectedObjectsMsg = bird_repellent_msgs::msg::DetectedObjects;

class BirdDetectorComponent : public rclcpp::Node {
public:
    BirdDetectorComponent(const rclcpp::NodeOptions& options);
    ~BirdDetectorComponent() {};

private:
    void InitRosParam();
    void InitRosCommon();
    void InitCommonClass();

    void MakeDir(const std::string& dir);

    void CallbackImage(const sensor_msgs::msg::Image::ConstSharedPtr& msg);

    rclcpp::Publisher<DetectedObjectsMsg>::SharedPtr m_pub_bird_objects;

    image_transport::Publisher m_pub_bird_image;
    image_transport::Subscriber m_sub_camera_image;

    bool m_is_compressed;
    
    std::string m_label_text_path;
    hailo_param_t m_hailo_param;
    std::shared_ptr<DetectionProcessor> m_detector;

    bool m_debug_mode;
    bool m_save_image;
    int m_save_count = 0;
    std::string m_save_dir;
};

} // namespace bird_detector

#endif // BIRD_DETECTOR_COMPONENT_HPP_