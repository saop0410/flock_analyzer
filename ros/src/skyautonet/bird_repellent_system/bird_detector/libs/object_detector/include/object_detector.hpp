#ifndef OBJECT_DETECTOR_HPP_
#define OBJECT_DETECTOR_HPP_

/*** Include ***/
/* for general */
#include <iostream>
#include <stdexcept>

/* for OpenCV */
#include <opencv2/opencv.hpp>

/* for Hailo */
#include <hailo_helper.hpp>

namespace hailo {
namespace util {

typedef struct {
    int x;
    int y;
    int width;
    int height;
} bbox_t;

typedef struct {
    int class_id;
    float prob;
    bbox_t box;
} object_t;

}

class Yolov5 {
public:
    Yolov5(std::string hef_path, float box_thres=0.25, bool show_info=true);

    Yolov5(const Yolov5&) = delete;
    Yolov5(Yolov5&& other) noexcept;

    ~Yolov5() {};

    Yolov5& operator=(const Yolov5&) = delete;
    Yolov5& operator=(Yolov5&& other) noexcept;
    
    std::vector<util::object_t> infer(util::image_ptr image_ptr, float image_width=0.f, float image_height=0.f);
    cv::Size2f get_input_size(size_t idx=0);
    
private:
    std::vector<util::object_t> DecodeHailoNms(std::vector<uint8_t>& data, hailo_nms_shape_t nms_info,
                                               float image_width, float image_height);

    float m_box_thres;
    std::vector<std::vector<uint8_t>> m_model_outputs;
    std::unique_ptr<hailo::HailoInferenceEngine> m_hailo_network;
};

}

#endif