#include "object_detector.hpp"

/*** Macro ***/
#define TAG "yolov5" //** 학습된 yolov5 사용하여 추론 수행하는 코드

using namespace hailo;
using namespace util;

Yolov5::Yolov5(std::string hef_path, float box_thres, bool show_info) : m_box_thres(box_thres) {
    m_hailo_network = std::make_unique<HailoInferenceEngine>(hef_path, false, show_info);

    m_model_outputs.resize(m_hailo_network->get_output_batch_size());
    for (size_t idx = 0; idx < m_hailo_network->get_output_batch_size(); idx++) {
        hailo_3d_image_shape_t output_info = m_hailo_network->get_output_shape(idx);
        size_t output_size = output_info.features * output_info.height * output_info.width;
        m_model_outputs[idx].resize(output_size * sizeof(uint8_t));
    }
}

Yolov5::Yolov5(Yolov5&& other) noexcept : m_box_thres(other.m_box_thres) {
    m_hailo_network = std::move(other.m_hailo_network);
    m_model_outputs.resize(m_hailo_network->get_output_batch_size());
    for (size_t idx = 0; idx < m_hailo_network->get_output_batch_size(); idx++) {
        hailo_3d_image_shape_t output_info = m_hailo_network->get_output_shape(idx);
        size_t output_size = output_info.features * output_info.height * output_info.width;
        m_model_outputs[idx].resize(output_size * sizeof(uint8_t));
    }

    other.m_box_thres = 0.f;
    other.m_model_outputs.clear();
    other.m_hailo_network.reset();
}

Yolov5& Yolov5::operator=(Yolov5&& other) noexcept {
    if (this != &other) {
        m_box_thres = other.m_box_thres;
        m_hailo_network = std::move(other.m_hailo_network);
        m_model_outputs.resize(m_hailo_network->get_output_batch_size());
        for (size_t idx = 0; idx < m_hailo_network->get_output_batch_size(); idx++) {
            hailo_3d_image_shape_t output_info = m_hailo_network->get_output_shape(idx);
            size_t output_size = output_info.features * output_info.height * output_info.width;
            m_model_outputs[idx].resize(output_size * sizeof(uint8_t));
        }

        other.m_box_thres = 0.f;
        other.m_model_outputs.clear();
        other.m_hailo_network.reset();
    }

    return *this;
}

std::vector<object_t> Yolov5::DecodeHailoNms(std::vector<uint8_t>& data, hailo_nms_shape_t nms_info,
                                             float image_width, float image_height)
{
    std::vector<object_t> objects;

    uint8_t* src_ptr = data.data();
    uint32_t actual_frame_size = 0;
    
    uint32_t num_of_classes = nms_info.number_of_classes;
    uint32_t max_bboxes_per_class = nms_info.max_bboxes_per_class;

    for (uint32_t class_index = 0; class_index < num_of_classes; class_index++) {
        float32_t bbox_count = *reinterpret_cast<const float32_t *>(src_ptr + actual_frame_size);

        if ((uint32_t)bbox_count > max_bboxes_per_class) {
            throw std::runtime_error(("Runtime error - Got more than the maximum bboxes per class in the nms buffer"));
        }

        if (bbox_count > 0) {
            uint8_t *class_ptr = src_ptr + actual_frame_size + sizeof(bbox_count);
            for (uint8_t box_index = 0; box_index < bbox_count; box_index++) {
                hailo_nms_bbox_t* bbox_struct = (hailo_nms_bbox_t*)(class_ptr + (box_index * sizeof(hailo_nms_bbox_t)));

                float confidence = clamp(bbox_struct->score, 0.0f, 1.0f);
                if (confidence > m_box_thres) {
                    float w = (float)(bbox_struct->x_max - bbox_struct->x_min) * image_width;
                    float h = (float)(bbox_struct->y_max - bbox_struct->y_min) * image_height;

                    bbox_t bbox = { 
                        static_cast<int>(bbox_struct->x_min * image_width),
                        static_cast<int>(bbox_struct->y_min * image_height),
                        static_cast<int>(w),
                        static_cast<int>(h)
                    };
                    
                    object_t object = { (int)class_index, confidence, bbox };
                    objects.push_back(object);
                }
            }
        }

        float32_t class_frame_size = static_cast<float32_t>(sizeof(bbox_count) + bbox_count * sizeof(hailo_nms_bbox_t));
        actual_frame_size += static_cast<uint32_t>(class_frame_size);
    }

    return objects;
}

std::vector<object_t> Yolov5::infer(image_ptr image_ptr, float image_width, float image_height) {
    m_hailo_network->infer(image_ptr, m_model_outputs);

    if (image_width <= 0.f) {
        image_width = m_hailo_network->get_input_shape().width;
    }

    if (image_height <= 0.f) {
        image_height = m_hailo_network->get_input_shape().height;
    }

    return DecodeHailoNms(m_model_outputs[0], m_hailo_network->get_nms_info(), image_width, image_height);
}

cv::Size2f Yolov5::get_input_size(size_t idx) {
    hailo_3d_image_shape_t input_shape = m_hailo_network->get_input_shape(idx);
    return cv::Size2f(input_shape.width, input_shape.height);
}