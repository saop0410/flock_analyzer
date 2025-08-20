#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/int32.hpp"
#include "std_msgs/msg/string.hpp"

#include <iostream>
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include <cstring>
#include <vector>

#define CAMERA_ID 0x02

class ZoomControlNode : public rclcpp::Node {
public:
    ZoomControlNode()
     : Node("zoom_control_node"), zoom_level_(declare_parameter<int>("initial_zoom", 0)),
     reset_on_start_(declare_parameter<bool>("reset_to_1x_on_start", true))
    {
        port_ = declare_parameter<std::string>("serial_port", "/dev/ttyUSB0");

        target_sub_ = this->create_subscription<std_msgs::msg::Int32>(
            "/zoom_control/target", 10,
            [this](std_msgs::msg::Int32::SharedPtr msg) {
                int target = msg->data;
                if (target < 0 || target >= static_cast<int>(zoom_times_.size())) {
                    RCLCPP_WARN(this->get_logger(), "Invalid target zoom: %d (Valid: 0~%lu)", target, zoom_times_.size() - 1);
                    return;
                }
                handleZoomToTarget(target);
            });

        current_pub_ = this->create_publisher<std_msgs::msg::Int32>("/zoom_control/current", 10);

        timer_ = this->create_wall_timer(std::chrono::seconds(1), [this]() {
            std_msgs::msg::Int32 msg;
            msg.data = zoom_level_ + 1;
            current_pub_->publish(msg);
        });

        fd_ = openSerialPort(port_.c_str());
        if (fd_ == -1) {
            RCLCPP_ERROR(this->get_logger(), "Failed to open serial port: %s", port_.c_str());
        } else {
            RCLCPP_INFO(this->get_logger(), "Serial port opened: %s", port_.c_str());
            if (reset_on_start_ && fd_ != -1) {
                RCLCPP_INFO(this->get_logger(), "Forcing full zoom out to minimum (level 0)...");
                sendPelcoDCommand(0x00, 0x40, 0x01, 0x00);
                usleep(zoom_times_.back());
                stop_zoom();
                zoom_level_ = 0;
            }
        }
    }

    ~ZoomControlNode() {
        if (fd_ != -1) close(fd_);
    }

private:
    int fd_;
    int zoom_level_;
    std::string port_;
    bool reset_on_start_;

    rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr target_sub_;
    rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr current_pub_;
    rclcpp::TimerBase::SharedPtr timer_;

    std::vector<int> zoom_times_ = {
        0,          // minimum zoom (logical x1)
        1000000,    // x2  | {data = 1}
        1500000,    // x3  | {data = 2}
        1700000,    // x4  | {data = 3}
        1900000,    // x5  | {data = 4}
        2050000,    // x6  | {data = 5}
        2150000,    // x7  | {data = 6}
        2260000,    // x8  | {data = 7}
        2360000,    // x9  | {data = 8}
        2400000,    // x10 | {data = 9}
        2510000,    // x11 | {data = 10}
        2570000,    // x12 | {data = 11}
        2640000,    // x13 | {data = 12}
        2700000,    // x14 | {data = 13}
        2770000,    // x15 | {data = 14}
        2890000,    // x16 | {data = 15}
        2980000,    // x17 | {data = 16}
        3140000,    // x18 | {data = 17}
        3250000,    // x19 | {data = 18}
        3400000,    // x20 | {data = 19}
        3540000,    // x21 | {data = 20}
        3650000,    // x22 | {data = 21}
        3770000,    // x23 | {data = 22}
        3900000,    // x24 | {data = 23}
        4020000,    // x25 | {data = 24}
        4150000,    // x26 | {data = 25}
        4270000,    // x27 | {data = 26}
        4400000,    // x28 | {data = 27}
        4500000,    // x29 | {data = 28}
        4600000     // x30 | {data = 29}
    };

    int openSerialPort(const char* port) {
        int fd = open(port, O_RDWR | O_NOCTTY | O_NDELAY);
        if (fd == -1) return -1;

        struct termios options;
        tcgetattr(fd, &options);
        cfsetispeed(&options, B9600);
        cfsetospeed(&options, B9600);

        options.c_cflag |= (CLOCAL | CREAD);
        options.c_cflag &= ~PARENB;
        options.c_cflag &= ~CSTOPB;
        options.c_cflag &= ~CSIZE;
        options.c_cflag |= CS8;
        options.c_lflag = 0;
        options.c_oflag = 0;
        options.c_iflag = 0;

        tcsetattr(fd, TCSANOW, &options);
        return fd;
    }

    void sendPelcoDCommand(uint8_t cmd1, uint8_t cmd2, uint8_t data1, uint8_t data2) {
        uint8_t packet[7];
        packet[0] = 0xFF;
        packet[1] = CAMERA_ID;
        packet[2] = cmd1;
        packet[3] = cmd2;
        packet[4] = data1;
        packet[5] = data2;
        packet[6] = (packet[1] + packet[2] + packet[3] + packet[4] + packet[5]) & 0xFF;
        write(fd_, packet, 7);
    }

    void stop_zoom() {
        sendPelcoDCommand(0x00, 0x00, 0x00, 0x00);
        usleep(50000);
    }

    void zoom_to_level(int from, int to) {
        if (from == to) return;

        int max_index = zoom_times_.size() - 1;
        if (from < 0 || from > max_index || to < 0 || to > max_index) {
            RCLCPP_WARN(this->get_logger(), "Zoom level out of range: from %d to %d", from, to);
            return;
        }

        int sleep_time = std::abs(zoom_times_[to] - zoom_times_[from]);

        if (from < to) {
            sendPelcoDCommand(0x00, 0x20, 0x01, 0x00);  // Zoom In
        } else {
            sendPelcoDCommand(0x00, 0x40, 0x01, 0x00);  // Zoom Out
        }

        usleep(sleep_time);
        stop_zoom();
    }

    void handleZoomToTarget(int target) {
        if (zoom_level_ == target) {
            RCLCPP_INFO(this->get_logger(), "Zoom already at target level: %d", target + 1);
            return;
        }
        RCLCPP_INFO(this->get_logger(), "Zooming from level %d to level %d", zoom_level_ + 1, target + 1);
        zoom_to_level(zoom_level_, target);
        zoom_level_ = target;
    }
};

int main(int argc, char * argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ZoomControlNode>());
    rclcpp::shutdown();
    return 0;
}