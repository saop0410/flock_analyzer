#include "flock_analyzer.hpp"
#include <opencv2/ximgproc.hpp>
#include <numeric> // For std::accumulate
#include <algorithm> // For std::min_element, std::max_element
#include <cmath> // For std::abs
#include <set>   // For species_set
#include <limits> // For numeric_limits

namespace flock_analyzer
{

// Helper function to find intersection of two lines (defined by two points each)
cv::Point2f _find_line_intersection(cv::Vec4i line1, cv::Vec4i line2)
{
    float x1 = line1[0], y1 = line1[1], x2 = line1[2], y2 = line1[3];
    float x3 = line2[0], y3 = line2[1], x4 = line2[2], y4 = line2[3];

    float den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
    if (std::abs(den) < 1e-6) { // Lines are parallel or collinear
        return cv::Point2f(-1, -1); // Return an invalid point
    }

    float t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den;
    
    float px = x1 + t * (x2 - x1);
    float py = y1 + t * (y2 - y1);
    return cv::Point2f(px, py);
}

// Constants from Python code for pattern image creation
const int GAUSSIAN_SIGMA_CRITICAL_POINTS = 3;
const int POINT_RADIUS_CRITICAL_POINTS = 2;

// Helper to create pattern image for critical points (grayscale heatmap)
cv::Mat FlockAnalyzer::_create_pattern_image_for_critical_points(const std::vector<cv::Point2f> &center_points, int original_w, int original_h, const cv::Size &target_img_size)
{
    cv::Mat img = cv::Mat::zeros(target_img_size, CV_8UC1); // Grayscale, black background

    float scale_x = static_cast<float>(target_img_size.width) / original_w;
    float scale_y = static_cast<float>(target_img_size.height) / original_h;

    for (const auto &p : center_points) {
        int x = static_cast<int>(p.x * scale_x);
        int y = static_cast<int>(p.y * scale_y);

        // Ensure points are within bounds
        if (x >= 0 && x < target_img_size.width && y >= 0 && y < target_img_size.height) {
            cv::circle(img, cv::Point(x, y), POINT_RADIUS_CRITICAL_POINTS, cv::Scalar(255), -1); // White circle
        }
    }

    // Apply Gaussian blur to create heatmap
    int ksize = static_cast<int>(6 * GAUSSIAN_SIGMA_CRITICAL_POINTS + 1);
    if (ksize % 2 == 0) ksize += 1;
    cv::GaussianBlur(img, img, cv::Size(ksize, ksize), GAUSSIAN_SIGMA_CRITICAL_POINTS);

    return img;
}


FlockAnalyzer::FlockAnalyzer(const std::string &vit_model_path, const std::string &vit_class_names_path)
{
    // classifier_ = std::make_shared<FormationClassifier>(vit_model_path, vit_class_names_path);
}

bird_repellent_msgs::msg::FlockAnalysis FlockAnalyzer::analyze(const bird_repellent_msgs::msg::DetectedObjects & msg)
{
    auto result = bird_repellent_msgs::msg::FlockAnalysis();
    result.header = msg.header;
    result.image_width = msg.image_width;
    result.image_height = msg.image_height;
    result.objects = msg.objects;
    result.number_of_birds = msg.objects.size();
    result.is_flock = result.number_of_birds > 5; // Python logic: > 5 birds for flock

    if (result.number_of_birds == 0) {
        previous_center_points_.clear();
        return result;
    }

    std::vector<cv::Point2f> current_center_points;
    double total_bird_area = 0;
    std::set<std::string> species_set;

    for (const auto& obj : msg.objects)
    {
        current_center_points.emplace_back(obj.box.x + obj.box.width / 2.0f, obj.box.y + obj.box.height / 2.0f);
        total_bird_area += obj.box.width * obj.box.height;
        species_set.insert(obj.species);
    }

    result.species.assign(species_set.begin(), species_set.end());

    // Classify formation
    if (result.number_of_birds > 1) { // Only classify formation if more than 1 bird
        // result.formation = classifier_->classify(current_center_points, msg.image_width, msg.image_height);
        result.formation = "unknown";
    } else {
        result.formation = "single"; // Or "unknown" if preferred for single birds
    }
    
    // Calculate density
    result.density = calculate_density(msg.objects, msg.image_width, msg.image_height);

    // Calculate number of clusters (simplified DBSCAN logic)
    if (result.number_of_birds > 2) {
        // Simplified clustering: if birds are very close, consider them one cluster.
        // This is a heuristic, not a full DBSCAN.
        // For a more robust solution, a C++ DBSCAN implementation or library would be needed.
        // For now, if there are more than 2 birds, we assume they form one cluster
        // unless they are very far apart (which is not explicitly checked here).
        // The Python code uses DBSCAN with eps=w*0.05.
        // For simplicity, if number of birds > 2, we'll assume 1 cluster for now,
        // or if they are very spread out, we could count them individually.
        // Let's stick to the Python logic's spirit: if enough birds, try to cluster.
        // If the formation is "dispersed", it implies multiple clusters or spread out.
        // For other formations, it's likely one cluster.
        if (result.formation == "dispersed") {
            // A very basic heuristic for dispersed: count individual birds as clusters
            // if they are not extremely close. This is a placeholder.
            result.number_of_clusters = result.number_of_birds;
        } else {
            result.number_of_clusters = 1; // Assume one cluster for other formations
        }
    } else {
        result.number_of_clusters = result.number_of_birds;
    }


    // Create pattern image for critical points calculation (especially for V-shape)
    cv::Mat pattern_image_for_cp = _create_pattern_image_for_critical_points(current_center_points, msg.image_width, msg.image_height);

    // Calculate critical points based on formation
    result.critical_points = calculate_critical_points(
        result.formation,
        current_center_points,
        msg.image_width,
        msg.image_height,
        pattern_image_for_cp
    );

    // Calculate direction
    result.direction = calculate_direction(current_center_points);
    previous_center_points_ = current_center_points; // Update previous points

    return result;
}

bird_repellent_msgs::msg::Direction FlockAnalyzer::calculate_direction(const std::vector<cv::Point2f> &current_center_points)
{
    bird_repellent_msgs::msg::Direction direction_vector;
    direction_vector.x = 0.0;
    direction_vector.y = 0.0;
    direction_vector.z = 0.0;

    if (previous_center_points_.empty() || current_center_points.empty()) {
        return direction_vector;
    }

    float prev_avg_x = 0.0f, prev_avg_y = 0.0f;
    for (const auto& p : previous_center_points_) {
        prev_avg_x += p.x;
        prev_avg_y += p.y;
    }
    prev_avg_x /= previous_center_points_.size();
    prev_avg_y /= previous_center_points_.size();

    float curr_avg_x = 0.0f, curr_avg_y = 0.0f;
    for (const auto& p : current_center_points) {
        curr_avg_x += p.x;
        curr_avg_y += p.y;
    }
    curr_avg_x /= current_center_points.size();
    curr_avg_y /= current_center_points.size();

    float delta_x = curr_avg_x - prev_avg_x;
    float delta_y = curr_avg_y - prev_avg_y;

    if (std::abs(delta_x) < 5 && std::abs(delta_y) < 5) {
        return direction_vector;
    }

    direction_vector.x = delta_x;
    direction_vector.y = delta_y;
    
    return direction_vector;
}


std::vector<geometry_msgs::msg::Point> FlockAnalyzer::calculate_critical_points(
    const std::string& formation,
    const std::vector<cv::Point2f>& center_points,
    int image_width,
    int image_height,
    const cv::Mat& pattern_image) // pattern_image is now passed
{
    std::vector<geometry_msgs::msg::Point> critical_points_msg;
    if (center_points.empty()) {
        return critical_points_msg;
    }

    if (formation == "V-shape") {
        // Use the passed pattern_image for skeletonization and line detection
        cv::Mat binary_pattern_img;
        // Binarize the pattern image based on brightness value 40 (from Python)
        cv::threshold(pattern_image, binary_pattern_img, 40, 255, cv::THRESH_BINARY);
        
        cv::Mat skeleton;
        // OpenCV's thinning requires CV_8UC1 binary image (0 or 255)
        cv::ximgproc::thinning(binary_pattern_img, skeleton);

        std::vector<cv::Vec4i> lines;
        // Parameters from Python: threshold=20, minLineLength=10, maxLineGap=5
        cv::HoughLinesP(skeleton, lines, 1, CV_PI / 180, 20, 10, 5);

        cv::Point2f apex_point(-1, -1);
        float min_y = std::numeric_limits<float>::max(); // Initialize with max float value

        if (lines.size() >= 2)
        {
            for (size_t i = 0; i < lines.size(); i++)
            {
                for (size_t j = i + 1; j < lines.size(); j++)
                {
                    cv::Point2f intersection = _find_line_intersection(lines[i], lines[j]);
                    if (intersection.x != -1 && intersection.y != -1) // Check for valid intersection
                    {
                        // Filter intersection points to be within image bounds (pattern image bounds)
                        if (intersection.x >= 0 && intersection.x < pattern_image.cols &&
                            intersection.y >= 0 && intersection.y < pattern_image.rows)
                        {
                            if (intersection.y < min_y) // For V-shape, apex is the lowest point
                            {
                                min_y = intersection.y;
                                apex_point = intersection;
                            }
                        }
                    }
                }
            }
        }
        
        if (apex_point.x != -1) {
            // Scale coordinates from pattern size (64x64) to original image size
            cv::Size pattern_size(64, 64); // Hardcoded from Python
            float scale_x = static_cast<float>(image_width) / pattern_size.width;
            float scale_y = static_cast<float>(image_height) / pattern_size.height;
            
            geometry_msgs::msg::Point p;
            p.x = apex_point.x * scale_x;
            p.y = apex_point.y * scale_y;
            p.z = 0.0;
            critical_points_msg.push_back(p);
        }

    } else if (formation == "Line") {
        if (center_points.size() >= 2) {
            double max_dist = 0;
            cv::Point2f p1_line, p2_line;

            for (size_t i = 0; i < center_points.size(); ++i) {
                for (size_t j = i + 1; j < center_points.size(); ++j) {
                    double dist = cv::norm(center_points[i] - center_points[j]);
                    if (dist > max_dist) {
                        max_dist = dist;
                        p1_line = center_points[i];
                        p2_line = center_points[j];
                    }
                }
            }
            geometry_msgs::msg::Point p; 
            p.x = p1_line.x; p.y = p1_line.y; p.z = 0.0;
            critical_points_msg.push_back(p);
            p.x = p2_line.x; p.y = p2_line.y; p.z = 0.0;
            critical_points_msg.push_back(p);
        } else if (center_points.size() == 1) {
            geometry_msgs::msg::Point p;
            p.x = center_points[0].x; p.y = center_points[0].y; p.z = 0.0;
            critical_points_msg.push_back(p);
        }

    } else if (formation == "dispersed") {
        // Python logic for dispersed critical points: centroid of largest cluster (DBSCAN)
        // or overall centroid if no significant clusters.
        // For C++, we'll use the overall centroid for simplicity as full DBSCAN is complex.
        if (center_points.size() >= 1) {
            double sum_x = 0, sum_y = 0;
            for (const auto& pt : center_points) {
                sum_x += pt.x;
                sum_y += pt.y;
            }
            geometry_msgs::msg::Point p;
            p.x = sum_x / center_points.size();
            p.y = sum_y / center_points.size();
            p.z = 0.0;
            critical_points_msg.push_back(p);
        }

    } else if (formation == "Ring") {
        // Python logic for Ring critical points: centroid of all points.
        if (center_points.size() >= 1) {
            double sum_x = 0, sum_y = 0;
            for (const auto& pt : center_points) {
                sum_x += pt.x;
                sum_y += pt.y;
            }
            geometry_msgs::msg::Point p;
            p.x = sum_x / center_points.size();
            p.y = sum_y / center_points.size();
            p.z = 0.0;
            critical_points_msg.push_back(p);
        }
    }
    // Add other formation types as needed

    return critical_points_msg;
}

float FlockAnalyzer::calculate_density(const std::vector<bird_repellent_msgs::msg::DetectedObject> &objects, int image_width, int image_height)
{
    if (objects.empty() || image_width == 0 || image_height == 0) {
        return 0.0f;
    }

    double total_area = 0;
    for (const auto& obj : objects) {
        total_area += obj.box.width * obj.box.height;
    }

    return static_cast<float>(total_area / (image_width * image_height));
}

} // namespace flock_analyzer