#ifndef HAILO_HELPER_HPP_
#define HAILO_HELPER_HPP_

/*** Include ***/
/* for general */
#include <iostream>

/* for Hailo */
#include <hailo/hailort.hpp>

namespace hailo {
namespace util {

using image_ptr = uint8_t*;

typedef struct {
    float32_t y_min;
    float32_t x_min;
    float32_t y_max;
    float32_t x_max;
    float32_t score;
} hailo_nms_bbox_t;

inline static uint8_t quant(float value, float qp_scale, float qp_zp) {
    return (uint8_t)((value / qp_scale) + qp_zp);
}

inline static float dequant(uint8_t value, float qp_scale, float qp_zp) {
    return (float)(((float)value - qp_zp) * qp_scale);
}

inline static float clamp(float val, float min, float max) {
    return val > min ? (val < max ? val : max) : min;
}

}

class HailoInferenceEngine {
public:
    HailoInferenceEngine(std::string hef_path,
                         bool multi_process=true, bool show_info=true);

    HailoInferenceEngine(const HailoInferenceEngine&) = delete;
    HailoInferenceEngine(HailoInferenceEngine&& other) noexcept;

    ~HailoInferenceEngine() {};

    HailoInferenceEngine& operator=(const HailoInferenceEngine&) = delete;
    HailoInferenceEngine& operator=(HailoInferenceEngine&& other) noexcept;

    void infer(util::image_ptr image_ptr, std::vector<std::vector<uint8_t>>& model_outputs);

    size_t get_input_batch_size();
    hailo_3d_image_shape_t get_input_shape(size_t idx=0);
    hailo_vstream_info_t get_intput_stream_info(size_t idx=0);

    size_t get_output_batch_size();
    hailo_3d_image_shape_t get_output_shape(size_t idx=0);
    hailo_vstream_info_t get_output_stream_info(size_t idx=0);
    
    hailo_nms_shape_t get_nms_info(size_t idx=0);
    
    hailo_quant_info_t get_quant_info(size_t idx=0);
    uint8_t quant(float value, size_t idx=0);
    float dequant(uint8_t value, size_t idx=0);

private:
    using ConfiguredNetworkGroup = hailort::Expected<std::shared_ptr<hailort::ConfiguredNetworkGroup>>;
    using VDevice = hailort::Expected<std::unique_ptr<hailort::VDevice>>;

    VDevice CreateVDevice(const bool multi_process);
    ConfiguredNetworkGroup ConfigureNetworkGroup(const std::string& hef_path, hailort::VDevice& device);
    hailo_status Initialize(const std::string& hef_path, const bool multi_process);

    hailo_status WriteHailo(hailort::InputVStream& input_vstream, util::image_ptr image_ptr);
    hailo_status ReadHailo(hailort::OutputVStream& output_vstream, std::vector<uint8_t>& model_output);

    std::unique_ptr<hailort::VDevice> m_vdevice;
    std::shared_ptr<hailort::ConfiguredNetworkGroup> m_network_group;

    std::vector<hailort::InputVStream> m_input_vstreams;
    std::vector<hailort::OutputVStream> m_output_vstreams;
};

}

#endif