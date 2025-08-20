#ifndef FLOCK_ANALYZER_FLOCK_ANALYZER_HPP
#define FLOCK_ANALYZER_FLOCK_ANALYZER_HPP

#include <rclcpp/rclcpp.hpp>
#include <bird_repellent_msgs/msg/detected_objects.hpp>
#include <bird_repellent_msgs/msg/flock_analysis.hpp>
#include <bird_repellent_msgs/msg/direction.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp> // For thinning
#include <vector>
#include <string>
#include <memory>
#include "formation_classifier.hpp"

namespace flock_analyzer
{

// Helper function to find intersection of two lines (defined by two points each)
cv::Point2f _find_line_intersection(cv::Vec4i line1, cv::Vec4i line2);

class FlockAnalyzer
{
public:
    FlockAnalyzer(const std::string &vit_model_path, const std::string &vit_class_names_path);

    bird_repellent_msgs::msg::FlockAnalysis analyze(const bird_repellent_msgs::msg::DetectedObjects &msg);

private:
    std::shared_ptr<FormationClassifier> classifier_;
    std::vector<cv::Point2f> previous_center_points_;

    float calculate_density(const std::vector<bird_repellent_msgs::msg::DetectedObject> &objects, int image_width, int image_height);
    bird_repellent_msgs::msg::Direction calculate_direction(const std::vector<cv::Point2f> &current_center_points);

    std::vector<geometry_msgs::msg::Point> calculate_critical_points(
        const std::string& formation,
        const std::vector<cv::Point2f>& center_points,
        int image_width,
        int image_height,
        const cv::Mat& pattern_image // Add pattern_image for V-shape analysis
    );

    // Helper to create pattern image for critical points (if needed, e.g., for V-shape skeletonization)
    cv::Mat _create_pattern_image_for_critical_points(const std::vector<cv::Point2f> &center_points, int original_w, int original_h, const cv::Size &target_img_size = cv::Size(64, 64));
};

} // namespace flock_analyzer

#endif // FLOCK_ANALYZER_FLOCK_ANALYZER_HPP