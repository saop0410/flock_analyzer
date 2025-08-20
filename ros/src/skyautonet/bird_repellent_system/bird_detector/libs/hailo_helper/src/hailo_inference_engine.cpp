#include "hailo_helper.hpp"
#include "hailo_infos.hpp"

/*** Macro ***/
#define TAG "HailoInferenceEngine"

using namespace hailo;

constexpr uint32_t DEVICE_COUNT = 1;
constexpr bool QUANTIZED = true;
constexpr hailo_format_type_t FORMAT_TYPE = HAILO_FORMAT_TYPE_AUTO;

HailoInferenceEngine::HailoInferenceEngine(std::string hef_path, bool multi_process, bool show_info) {
    std::cout << "\n======= Init Hailo =======" << std::endl;
    Initialize(hef_path, multi_process);

    if (show_info) {
        std::cout << "======= input info =======" << std::endl;
        std::vector<hailo_vstream_info_t> input_infos;
        for (auto& stream : m_input_vstreams) {
            input_infos.push_back(stream.get_info());
        }
        print_vstream_info(input_infos);

        std::cout << "====== output info =======" << std::endl;
        std::vector<hailo_vstream_info_t> output_infos;
        for (auto& stream : m_output_vstreams) {
            output_infos.push_back(stream.get_info());
        }
        print_vstream_info(output_infos);
        std::cout << "==========================" << std::endl;
    }
}

HailoInferenceEngine::HailoInferenceEngine(HailoInferenceEngine&& other) noexcept {
    m_vdevice = std::move(other.m_vdevice);
    m_network_group = std::move(other.m_network_group);

    m_input_vstreams.reserve(other.m_input_vstreams.size());
    for(auto& stream : other.m_input_vstreams) {
        m_input_vstreams.push_back(std::move(stream));
    }

    m_output_vstreams.reserve(other.m_output_vstreams.size());
    for(auto& stream : other.m_output_vstreams) {
        m_output_vstreams.push_back(std::move(stream));
    }

    other.m_network_group.reset();
    other.m_input_vstreams.clear();
    other.m_output_vstreams.clear();
}

HailoInferenceEngine& HailoInferenceEngine::operator=(HailoInferenceEngine&& other) noexcept {
    if (this != &other) {
        m_vdevice = std::move(other.m_vdevice);
        m_network_group = std::move(other.m_network_group);

        m_input_vstreams.reserve(other.m_input_vstreams.size());
        for(auto& stream : other.m_input_vstreams) {
            m_input_vstreams.push_back(std::move(stream));
        }

        m_output_vstreams.reserve(other.m_output_vstreams.size());
        for(auto& stream : other.m_output_vstreams) {
            m_output_vstreams.push_back(std::move(stream));
        }

        other.m_network_group.reset();
        other.m_input_vstreams.clear();
        other.m_output_vstreams.clear();
    }

    return *this;
}

HailoInferenceEngine::VDevice HailoInferenceEngine::CreateVDevice(const bool multi_process) {
    hailo_vdevice_params_t params;
    auto status = hailo_init_vdevice_params(&params);
    if (HAILO_SUCCESS != status) {
        std::cerr << "Failed init vdevice_params, status = " << status << std::endl;
        return hailort::make_unexpected(status);
    }

    if (multi_process) {
        params.device_count = DEVICE_COUNT;
        params.multi_process_service = true;
        params.group_id = "SHARED";
    }

    return hailort::VDevice::create(params);
}

HailoInferenceEngine::ConfiguredNetworkGroup HailoInferenceEngine::ConfigureNetworkGroup(
                                                    const std::string& hef_path, hailort::VDevice& vdevice)
{
    auto hef = hailort::Hef::create(hef_path);
    if (!hef) {
        return hailort::make_unexpected(hef.status());
    }

    auto configure_params = vdevice.create_configure_params(hef.value());
    if (!configure_params) {
        return hailort::make_unexpected(configure_params.status());
    }

    auto network_groups = vdevice.configure(hef.value(), configure_params.value());
    if (!network_groups) {
        return hailort::make_unexpected(network_groups.status());
    }

    if (network_groups->size() != 1) {
        std::cerr << "Invalid amount of network groups" << std::endl;
        return hailort::make_unexpected(HAILO_INTERNAL_FAILURE);
    }

    return std::move(network_groups->at(0));
}

hailo_status HailoInferenceEngine::Initialize(const std::string& hef_path, const bool multi_process) {
    auto vdevice_exp = CreateVDevice(multi_process);
    if (!vdevice_exp) {
        std::cerr << "Failed to create vdevice, status = " << vdevice_exp.status() << std::endl;
        return vdevice_exp.status();
    }
    m_vdevice = vdevice_exp.release();

    auto network_group_exp = ConfigureNetworkGroup(hef_path, *m_vdevice);
    if (!network_group_exp) {
        std::cerr << "Failed to configure network group " << hef_path << std::endl;
        return network_group_exp.status();
    }
    m_network_group = network_group_exp.release();

    auto vstreams = hailort::VStreamsBuilder::create_vstreams(*m_network_group, QUANTIZED, FORMAT_TYPE);
    if (!vstreams) {
        std::cerr << "Failed creating vstreams " << vstreams.status() << std::endl;
        return vstreams.status();
    }

    m_input_vstreams.reserve(vstreams->first.size());
    for (auto& vstream : vstreams->first) {
        m_input_vstreams.push_back(std::move(vstream));
    }

    m_output_vstreams.reserve(vstreams->second.size());
    for (auto& vstream : vstreams->second) {
        m_output_vstreams.push_back(std::move(vstream));
    }

    return HAILO_SUCCESS;
}

hailo_status HailoInferenceEngine::WriteHailo(hailort::InputVStream& input_vstream, util::image_ptr image_ptr) {
    size_t input_size = input_vstream.get_frame_size();
    hailo_status status = input_vstream.write(hailort::MemoryView(image_ptr, input_size));
    if (HAILO_SUCCESS != status) {
        return status;
    }

    return status;
}

hailo_status HailoInferenceEngine::ReadHailo(hailort::OutputVStream& output_vstream, std::vector<uint8_t>& model_output) {
    if (model_output.size() != output_vstream.get_frame_size()) {
        model_output.resize(output_vstream.get_frame_size());
    }

    hailo_status status = output_vstream.read(hailort::MemoryView(model_output.data(), model_output.size()));
    if (HAILO_SUCCESS != status) {
        return status;
    }

    return status;
}

void HailoInferenceEngine::infer(util::image_ptr image_ptr, std::vector<std::vector<uint8_t>>& model_outputs) {
    hailo_status status = HAILO_SUCCESS;

    for (auto& input_vstream : m_input_vstreams) {
        status = WriteHailo(input_vstream, image_ptr);
        if (status != HAILO_SUCCESS) {
            return;
        }
    }

    size_t output_size = m_output_vstreams.size();
    if(model_outputs.size() != output_size) {
        model_outputs.resize(output_size);
    }
    for (size_t idx=0; idx<output_size; idx++) {
        status = ReadHailo(m_output_vstreams[idx], model_outputs[idx]);
        if (status != HAILO_SUCCESS) {
            return;
        }
    }

    if (status != HAILO_SUCCESS) {
        std::cerr << "Inference error status : " << status << std::endl;
    }

    return;
}

size_t HailoInferenceEngine::get_input_batch_size() {
    return m_input_vstreams.size();
}

hailo_3d_image_shape_t HailoInferenceEngine::get_input_shape(size_t idx) {
    if (m_input_vstreams.size() > idx) {
        return m_input_vstreams[idx].get_info().shape;
    } else {
        throw std::out_of_range("get_input_shape: index out of range");
    }
}

hailo_vstream_info_t HailoInferenceEngine::get_intput_stream_info(size_t idx) {
    if (m_input_vstreams.size() > idx) {
        return m_input_vstreams[idx].get_info();
    } else {
        throw std::out_of_range("get_intput_stream_info: index out of range");
    }
}

size_t HailoInferenceEngine::get_output_batch_size() {
    return m_output_vstreams.size();
}

hailo_3d_image_shape_t HailoInferenceEngine::get_output_shape(size_t idx) {
    if (m_output_vstreams.size() > idx) {
        return m_output_vstreams[idx].get_info().shape;
    } else {
        throw std::out_of_range("get_output_shape: index out of range");
    }
}

hailo_vstream_info_t HailoInferenceEngine::get_output_stream_info(size_t idx) {
    if (m_output_vstreams.size() > idx) {
        return m_output_vstreams[idx].get_info();
    } else {
        throw std::out_of_range("get_output_stream_info: index out of range");
    }
}

hailo_nms_shape_t HailoInferenceEngine::get_nms_info(size_t idx) {
    if (m_output_vstreams.size() > idx) {
        return m_output_vstreams[idx].get_info().nms_shape;
    } else {
        throw std::out_of_range("get_nms_info: index out of range");
    }
}

hailo_quant_info_t HailoInferenceEngine::get_quant_info(size_t idx) {
    if (m_output_vstreams.size() > idx) {
        return m_output_vstreams[idx].get_info().quant_info;
    } else {
        throw std::out_of_range("get_quant_info: index out of range");
    }
}

uint8_t HailoInferenceEngine::quant(float value, size_t idx) {
    hailo_quant_info_t q_info = get_quant_info(idx);
    return util::quant(value, q_info.qp_scale, q_info.qp_zp);
}

float HailoInferenceEngine::dequant(uint8_t value, size_t idx) {
    hailo_quant_info_t q_info = get_quant_info(idx);
    return util::dequant(value, q_info.qp_scale, q_info.qp_zp);
}