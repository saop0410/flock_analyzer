#include <filesystem>

#include "bird_detector.hpp"

namespace bird_detector {

BirdDetectorComponent::BirdDetectorComponent(const rclcpp::NodeOptions& options)
 : Node("bird_detector", options)
{
    InitRosParam();
    InitRosCommon();
    InitCommonClass();
    RCLCPP_INFO(this->get_logger(), "bird_detector initialization");
}

void BirdDetectorComponent::InitRosParam() {
    std::string model_path = declare_parameter("model_path", "");
    
    // common param
    m_is_compressed = declare_parameter("is_compressed", false);
    
    // hailo model param
    m_label_text_path        = model_path + declare_parameter("label_text_path", "");
    m_hailo_param.model_path = model_path + declare_parameter("hailo_model", "");
    m_hailo_param.box_thres  = declare_parameter("box_thresh", 0.25);
    
    // debug param
    m_debug_mode = declare_parameter("debug_mode", false);
    m_save_image = declare_parameter("save_image", false);
    m_save_dir   = declare_parameter("save_image_path", "");

    // print param
    RCLCPP_INFO(this->get_logger(), "---------------- ros param -------------- ");
    RCLCPP_INFO(this->get_logger(), "-- common param --");
    RCLCPP_INFO(this->get_logger(), "is_compressed : %d\n", m_is_compressed);

    RCLCPP_INFO(this->get_logger(), "-- hailo model param --");
    RCLCPP_INFO(this->get_logger(), "label_text_path : %s", m_label_text_path.c_str());
    RCLCPP_INFO(this->get_logger(), "model_path      : %s", m_hailo_param.model_path.c_str());
    RCLCPP_INFO(this->get_logger(), "box_thres       : %f\n", m_hailo_param.box_thres);
    
    RCLCPP_INFO(this->get_logger(), "-- debug param --");
    RCLCPP_INFO(this->get_logger(), "debug_mode : %d", m_debug_mode);
    RCLCPP_INFO(this->get_logger(), "save_image : %d", m_save_image);
    RCLCPP_INFO(this->get_logger(), "save_dir   : %s", m_save_dir.c_str());
    RCLCPP_INFO(this->get_logger(), "---------------- ros param -------------- ");
}

void BirdDetectorComponent::InitRosCommon() {
    rclcpp::QoS qos_profile(10);
    qos_profile.reliability(rclcpp::ReliabilityPolicy::BestEffort);
    qos_profile.durability(rclcpp::DurabilityPolicy::Volatile);

    m_pub_bird_objects = this->create_publisher<DetectedObjectsMsg>("out/bird/info", qos_profile);

    if (m_debug_mode) {
        m_pub_bird_image = image_transport::create_publisher(this, "out/bird/image",
                                                             qos_profile.get_rmw_qos_profile());
    }

    std::string transport = m_is_compressed ? "compressed" : "raw";

    std::string node_namespace(this->get_namespace());
    if (!node_namespace.empty() && node_namespace.back() != '/') {
        node_namespace += "/";
    }

    m_sub_camera_image = image_transport::create_subscription(this, node_namespace + "in/image",
                                                              std::bind(&BirdDetectorComponent::CallbackImage, this, std::placeholders::_1),
                                                              transport, qos_profile.get_rmw_qos_profile());
}

void BirdDetectorComponent::InitCommonClass() {
    m_detector = std::make_shared<DetectionProcessor>(m_hailo_param, m_label_text_path);

    m_save_count = 0;
    if (m_debug_mode && m_save_image) {
        MakeDir(m_save_dir);
    }
}

void BirdDetectorComponent::MakeDir(const std::string& dir) {
    try {
        if (!std::filesystem::exists(dir)) {
            if (!std::filesystem::create_directories(dir)) {
                std::cerr << "make dir failed" << std::endl;
            }
        }
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "filesystem_error in MakeDir(): " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Exception in MakeDir(): " << e.what() << std::endl;
    }
}

void BirdDetectorComponent::CallbackImage(const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
    cv_bridge::CvImagePtr image_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    cv::Mat& image = image_ptr->image;

    std::vector<hailo::util::object_t> bird_objects = m_detector->hailo_process(image); //**실제 검출 수행 

    DetectedObjectsMsg msg_objects;
    msg_objects.header = msg->header;
    msg_objects.image_width = image.cols;
    msg_objects.image_height = image.rows;

    for (const auto& bird : bird_objects) {
        DetectedObjectMsg msg_object;

        msg_object.class_id   = bird.class_id;
        msg_object.prob       = bird.prob;
        msg_object.box.x      = bird.box.x;
        msg_object.box.y      = bird.box.y;
        msg_object.box.width  = bird.box.width;
        msg_object.box.height = bird.box.height;

        msg_objects.objects.emplace_back(msg_object);
    }
    
    m_pub_bird_objects->publish(msg_objects);
    
    if (m_debug_mode && m_pub_bird_image) {
        for (const auto& bird : bird_objects) {
            cv::Rect bbox(bird.box.x, bird.box.y, bird.box.width, bird.box.height);
            cv::rectangle(image, bbox, cv::Scalar(0, 0, 255), 2);
        }

        if (!bird_objects.empty() && m_save_image) {
            std::string dir(m_save_dir + "/image_" + std::to_string(m_save_count) + ".png");
            cv::imwrite(dir, image);
            m_save_count++;
        }

        m_pub_bird_image.publish(*image_ptr->toImageMsg());
    }
}

} // namespace bird_detector

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(bird_detector::BirdDetectorComponent)