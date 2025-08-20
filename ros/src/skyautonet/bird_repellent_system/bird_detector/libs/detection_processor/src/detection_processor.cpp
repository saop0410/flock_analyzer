/* for My modules */
#include "detection_processor.hpp"

/* for OpenCV */
#include <opencv2/imgproc.hpp>

/* for general */
#include <utility>
#include <fstream>
#include <sstream>

/*** Macro ***/
#define TAG "DetectionProcessor"

DetectionProcessor::DetectionProcessor(hailo_param_t hailo_param, std::string label_text_path) {
    m_hailo_engine = std::make_shared<hailo::Yolov5>(hailo_param.model_path, hailo_param.box_thres);
    ReadLabels(label_text_path);
    std::cout << "DetectionProcessor Hailo initialization" << std::endl;
}

bool DetectionProcessor::ReadLabels(const std::string& filename) {
    std::ifstream ifs(filename);
    if (ifs.fail()) {
        std::cout << "Failed to read " << filename << std::endl;
        return false;
    }

    std::string str;
    while (getline(ifs, str)) {
        m_labels.push_back(str);
    }

    return true;
}

cv::Size DetectionProcessor::PutTextInfo(const std::string str, cv::Mat& image, cv::Point point, int min_w) {
    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    double fontScale = 0.5;
    int thickness = 2;
    int lineType = cv::LINE_4;
    int baseline = 0;

    cv::Size text_size = cv::getTextSize(str, fontFace, fontScale, thickness, &baseline);
    text_size.height += baseline;
    if (text_size.width < min_w) {
        text_size.width = min_w;
    }

    cv::Rect text_rect(point, text_size);
    cv::rectangle(image, text_rect, m_color.black, -1, lineType);

    point.y += (text_size.height - baseline);
    cv::putText(image, str, point, fontFace, fontScale, m_color.white, thickness);

    return text_size;
}

std::vector<hailo::util::object_t> DetectionProcessor::hailo_process(cv::Mat& image) {
    if (!m_hailo_engine) {
        std::cout << "Not Hailo Device Initialized" << std::endl;
        return std::vector<hailo::util::object_t>();
    }

    cv::Mat input_image;
    cv::cvtColor(image, input_image, cv::COLOR_BGR2RGB);
    cv::resize(input_image, input_image, m_hailo_engine->get_input_size());

    return m_hailo_engine->infer(input_image.data, image.cols, image.rows);
}