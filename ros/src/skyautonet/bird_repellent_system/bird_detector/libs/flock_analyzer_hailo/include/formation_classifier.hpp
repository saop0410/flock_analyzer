#ifndef FLOCK_ANALYZER_FORMATION_CLASSIFIER_HPP
#define FLOCK_ANALYZER_FORMATION_CLASSIFIER_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <memory>
#include <fstream> // For reading class names

// Hailo SDK includes
#include <hailo/hailort.hpp>
#include <hailo/hailort_common.hpp>

namespace flock_analyzer
{

class FormationClassifier
{
public:
    FormationClassifier(const std::string &model_path, const std::string &class_names_path);
    std::string classify(const std::vector<cv::Point2f> &points, int original_w, int original_h);

private:
    std::unique_ptr<hailo_rt::ConfiguredNetworkGroup> m_configured_network_group;
    std::vector<hailo_rt::VStreamPtr> m_input_vstreams;
    std::vector<hailo_rt::VStreamPtr> m_output_vstreams;
    std::vector<std::string> class_names_;

    // Helper to create pattern image (grayscale heatmap)
    cv::Mat _create_pattern_image(const std::vector<cv::Point2f> &center_points, int original_w, int original_h, const cv::Size &target_img_size = cv::Size(64, 64));
};

} // namespace flock_analyzer

#endif // FLOCK_ANALYZER_FORMATION_CLASSIFIER_HPP