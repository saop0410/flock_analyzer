#include <rclcpp/rclcpp.hpp>
#include "flock_analyzer.hpp"
#include <bird_repellent_msgs/msg/detected_objects.hpp>
#include <bird_repellent_msgs/msg/flock_analysis.hpp>

class FlockAnalyzerROSNode : public rclcpp::Node
{
public:
    FlockAnalyzerROSNode()
    : Node("flock_analyzer_node")
    {
        // Declare parameters for model paths and dataset root
        this->declare_parameter<std::string>("vit_model_path", "");
        this->declare_parameter<std::string>("vit_class_names_path", "");

        std::string vit_model_path = this->get_parameter("vit_model_path").as_string();
        std::string vit_class_names_path = this->get_parameter("vit_class_names_path").as_string();

        RCLCPP_INFO(this->get_logger(), "ViT Model Path: %s", vit_model_path.c_str());
        RCLCPP_INFO(this->get_logger(), "ViT Class Names Path: %s", vit_class_names_path.c_str());

        // analyzer_ = std::make_shared<flock_analyzer::FlockAnalyzer>(vit_model_path, vit_class_names_path);
        analyzer_ = std::make_shared<flock_analyzer::FlockAnalyzer>("", "");

        subscription_ = this->create_subscription<bird_repellent_msgs::msg::DetectedObjects>(
            "detected_objects",
            10,
            std::bind(&FlockAnalyzerROSNode::detected_objects_callback, this, std::placeholders::_1)
        );

        publisher_ = this->create_publisher<bird_repellent_msgs::msg::FlockAnalysis>("flock_analysis", 10);

        RCLCPP_INFO(this->get_logger(), "Flock Analyzer ROS Node initialized.");
    }

private:
    void detected_objects_callback(const bird_repellent_msgs::msg::DetectedObjects::SharedPtr msg)
    {
        RCLCPP_INFO(this->get_logger(), "Received DetectedObjects message.");
        bird_repellent_msgs::msg::FlockAnalysis analysis_result = analyzer_->analyze(*msg);
        publisher_->publish(analysis_result);
        RCLCPP_INFO(this->get_logger(), "Published FlockAnalysis message.");
    }

    std::shared_ptr<flock_analyzer::FlockAnalyzer> analyzer_;
    rclcpp::Subscription<bird_repellent_msgs::msg::DetectedObjects>::SharedPtr subscription_;
    rclcpp::Publisher<bird_repellent_msgs::msg::FlockAnalysis>::SharedPtr publisher_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<FlockAnalyzerROSNode>());
    rclcpp::shutdown();
    return 0;
}