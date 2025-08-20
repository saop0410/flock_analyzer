#include "formation_classifier.hpp"
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <numeric> // For std::iota

// Hailo SDK includes
#include <hailo/hailort.hpp>
#include <hailo/hailort_common.hpp>

namespace flock_analyzer
{

// Constants from Python code
const int GAUSSIAN_SIGMA = 3;
const int POINT_RADIUS = 2;

FormationClassifier::FormationClassifier(const std::string &model_path, const std::string &class_names_path)
{
    try {
        // Load class names from file
        std::ifstream file(class_names_path);
        if (file.is_open()) {
            std::string line;
            while (std::getline(file, line)) {
                class_names_.push_back(line);
            }
            file.close();
            std::cout << "Loaded " << class_names_.size() << " class names from " << class_names_path << std::endl;
        } else {
            std::cerr << "Error: Unable to open class names file: " << class_names_path << std::endl;
            throw std::runtime_error("Unable to open class names file.");
        }

        // Initialize HailoRT device and load HEF
        auto device = hailo_rt::Device::create();
        if (!device.has_value()) {
            std::cerr << "Failed to create HailoRT device: " << device.status().message() << std::endl;
            throw std::runtime_error("Failed to create HailoRT device.");
        }

        auto hef = hailo_rt::Hef::create(model_path);
        if (!hef.has_value()) {
            std::cerr << "Failed to create HEF from " << model_path << ": " << hef.status().message() << std::endl;
            throw std::runtime_error("Failed to create HEF.");
        }

        auto configure_params = hailo_rt::ConfigureParams::create(hef.value());
        if (!configure_params.has_value()) {
            std::cerr << "Failed to create configure params: " << configure_params.status().message() << std::endl;
            throw std::runtime_error("Failed to create configure params.");
        }

        auto network_groups = hailo_rt::VStreamsBuilder::create_vstream_infos(hef.value(), configure_params.value().network_groups_params);
        if (!network_groups.has_value() || network_groups.value().empty()) {
            std::cerr << "Failed to get network groups from HEF: " << network_groups.status().message() << std::endl;
            throw std::runtime_error("Failed to get network groups.");
        }

        auto configured_network_group = hailo_rt::ConfiguredNetworkGroup::create(device.value(), hef.value(), network_groups.value().begin()->first, configure_params.value().network_groups_params.begin()->second);
        if (!configured_network_group.has_value()) {
            std::cerr << "Failed to configure network group: " << configured_network_group.status().message() << std::endl;
            throw std::runtime_error("Failed to configure network group.");
        }
        m_configured_network_group = std::move(configured_network_group.value());

        auto input_vstream_infos = m_configured_network_group->get_input_vstream_infos();
        auto output_vstream_infos = m_configured_network_group->get_output_vstream_infos();

        auto input_vstreams = hailo_rt::VStreamsBuilder::create_vstreams(*m_configured_network_group, input_vstream_infos);
        if (!input_vstreams.has_value()) {
            std::cerr << "Failed to create input vstreams: " << input_vstreams.status().message() << std::endl;
            throw std::runtime_error("Failed to create input vstreams.");
        }
        m_input_vstreams = std::move(input_vstreams.value());

        auto output_vstreams = hailo_rt::VStreamsBuilder::create_vstreams(*m_configured_network_group, output_vstream_infos);
        if (!output_vstreams.has_value()) {
            std::cerr << "Failed to create output vstreams: " << output_vstreams.status().message() << std::endl;
            throw std::runtime_error("Failed to create output vstreams.");
        }
        m_output_vstreams = std::move(output_vstreams.value());

        std::cout << "Hailo model loaded successfully from " << model_path << std::endl;

    } catch (const std::exception &e) {
        std::cerr << "Error initializing FormationClassifier with Hailo: " << e.what() << std::endl;
        throw; // Re-throw to indicate failure
    }
}

cv::Mat FormationClassifier::_create_pattern_image(const std::vector<cv::Point2f> &center_points, int original_w, int original_h, const cv::Size &target_img_size)
{
    cv::Mat img = cv::Mat::zeros(target_img_size, CV_8UC1); // Grayscale, black background

    float scale_x = static_cast<float>(target_img_size.width) / original_w;
    float scale_y = static_cast<float>(target_img_size.height) / original_h;

    for (const auto &p : center_points) {
        int x = static_cast<int>(p.x * scale_x);
        int y = static_cast<int>(p.y * scale_y);

        // Ensure points are within bounds
        if (x >= 0 && x < target_img_size.width && y >= 0 && y < target_img_size.height) {
            cv::circle(img, cv::Point(x, y), POINT_RADIUS, cv::Scalar(255), -1); // White circle
        }
    }

    // Apply Gaussian blur to create heatmap
    int ksize = static_cast<int>(6 * GAUSSIAN_SIGMA + 1);
    if (ksize % 2 == 0) ksize += 1;
    cv::GaussianBlur(img, img, cv::Size(ksize, ksize), GAUSSIAN_SIGMA);

    return img;
}

std::string FormationClassifier::classify(const std::vector<cv::Point2f> &points, int original_w, int original_h)
{
    if (points.empty()) {
        return "unknown";
    }

    // Create pattern image from center points
    cv::Mat pattern_image = _create_pattern_image(points, original_w, original_h);

    // Convert to float and normalize to [0, 1]
    // Assuming Hailo model expects float32 input in range [0, 1] or [0, 255]
    // Adjust this based on your HEF's input requirements
    pattern_image.convertTo(pattern_image, CV_32F, 1.0 / 255.0); // Normalize to [0, 1]

    // Prepare input for Hailo
    // Assuming single input vstream and input shape matches pattern_image (e.g., 1x64x64)
    if (m_input_vstreams.empty()) {
        std::cerr << "Error: No input vstreams available." << std::endl;
        return "unknown";
    }
    hailo_rt::VStream &input_vstream = *m_input_vstreams[0];
    hailo_rt::VStream &output_vstream = *m_output_vstreams[0]; // Assuming single output vstream

    // Get input buffer size
    size_t input_buffer_size = input_vstream.get_frame_size();
    std::vector<uint8_t> input_buffer(input_buffer_size);

    // Copy pattern_image data to input_buffer
    // This assumes the input_buffer is a flat array and pattern_image is grayscale
    // If your Hailo model expects a different layout (e.g., CHW for RGB),
    // you'll need to adjust this copy and potentially add channel handling.
    if (pattern_image.isContinuous()) {
        memcpy(input_buffer.data(), pattern_image.data, input_buffer_size);
    } else {
        // Handle non-continuous case if necessary, though _create_pattern_image should produce continuous
        std::cerr << "Warning: pattern_image is not continuous. Manual copy needed." << std::endl;
        for (int i = 0; i < pattern_image.rows; ++i) {
            memcpy(input_buffer.data() + i * pattern_image.cols * sizeof(float),
                   pattern_image.ptr<float>(i),
                   pattern_image.cols * sizeof(float));
        }
    }

    // Write input to Hailo
    auto input_status = input_vstream.write(hailo_rt::MemoryView(input_buffer.data(), input_buffer_size));
    if (!input_status.ok()) {
        std::cerr << "Failed to write to input vstream: " << input_status.message() << std::endl;
        return "unknown";
    }

    // Read output from Hailo
    size_t output_buffer_size = output_vstream.get_frame_size();
    std::vector<uint8_t> output_buffer(output_buffer_size);
    auto output_status = output_vstream.read(hailo_rt::MemoryView(output_buffer.data(), output_buffer_size));
    if (!output_status.ok()) {
        std::cerr << "Failed to read from output vstream: " << output_status.message() << std::endl;
        return "unknown";
    }

    // Process output (assuming float32 output and single class prediction)
    // This part depends heavily on your ViT model's output layer.
    // Assuming a classification output where the largest value corresponds to the class index.
    const float* output_data = reinterpret_cast<const float*>(output_buffer.data());
    int max_index = -1;
    float max_val = -1.0f;

    // Assuming output_buffer contains probabilities for each class
    // You need to know the number of output classes from your model
    // For example, if you have N classes, the output_buffer will contain N float values.
    // Replace `NUM_CLASSES` with the actual number of classes your ViT model outputs.
    const int NUM_CLASSES = class_names_.size(); // Assuming class_names_ size matches model output classes

    for (int i = 0; i < NUM_CLASSES; ++i) {
        if (output_data[i] > max_val) {
            max_val = output_data[i];
            max_index = i;
        }
    }

    if (max_index >= 0 && max_index < class_names_.size()) {
        return class_names_[max_index];
    } else {
        return "unknown";
    }
}

} // namespace flock_analyzer