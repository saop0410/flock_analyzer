#ifndef DETECTION_PROCESSOR_HPP_
#define DETECTION_PROCESSOR_HPP_

/*** Include ***/
/* for general */
#include <iostream>
#include <vector>
#include <string>

/* for OpenCV */
#include <opencv2/core.hpp>

/* for My modules */
#include "object_detector.hpp"

typedef struct {
    std::string model_path;
    float box_thres;
} hailo_param_t;

class DetectionProcessor {
public:
    DetectionProcessor(hailo_param_t hailo_param, std::string label_text_path);
    ~DetectionProcessor() {};

    cv::Rect bird_roi_cluster(const hailo::util::bbox_t& bbox, const cv::Size image_size,
                              int min_pixel=64, int padding=0);

    std::vector<hailo::util::object_t> hailo_process(cv::Mat& image);
    std::pair<std::vector<hailo::util::object_t>, bool> debug_process(const cv::Mat& image, cv::Rect roi, cv::Mat& debug);

private:
    typedef struct {
        const cv::Scalar red    = cv::Scalar( 50,  50, 200);
        const cv::Scalar blue   = cv::Scalar(200,  50,  50);
        const cv::Scalar green  = cv::Scalar( 50, 200,  50);
        const cv::Scalar yellow = cv::Scalar( 50, 200, 200);
        const cv::Scalar gray   = cv::Scalar(200, 200, 200);
        const cv::Scalar black  = cv::Scalar( 50,  50,  50);
        const cv::Scalar white  = cv::Scalar(250, 250, 250);
    } cv_color_t;

    bool ReadLabels(const std::string& filename);
    cv::Size PutTextInfo(const std::string str, cv::Mat& image, cv::Point point, int min_w=150);
    
    std::vector<std::string> m_labels;
    std::shared_ptr<hailo::Yolov5> m_hailo_engine;
    const cv_color_t m_color;
};

#endif