# skyautonet\_birdro

## Name

**skyautonet\_birdro**

## Description

**skyautonet\_birdro** is an innovative **autonomous mobile robot system** designed for intelligent **bird deterrence and control**. Leveraging advanced **autonomous driving capabilities** and specialized **bird detection and tracking modules**, this system aims to provide an effective and automated solution for mitigating bird-related issues in various environments, such as agriculture, airports, or industrial facilities.

Our system combines sophisticated robotics with intelligent vision and control to minimize human intervention while maximizing the efficiency of bird dispersal.

-----

### Features

  * **Autonomous Navigation:** Self-driving capabilities including localization, planning, and control for efficient movement in diverse terrains.
  * **Real-time Bird Detection & Tracking:** Utilizes camera and advanced algorithms for accurate identification and continuous tracking of birds.
  * **Multi-modal Bird Deterrence:** Employs various methods like **laser, lamp, and sound (speaker control)** for effective and humane bird dispersal.
  * **Modular ROS2 Architecture:** Built on ROS2 for flexible development, integration, and scalability.
  * **Integrated Hardware Control:** Direct management of robotic actuators and deterrent mechanisms via GPIO.

-----

## Visuals

*(Once you have them, you can add screenshots or GIFs here to showcase the robot in action, the bird detection interface, or the autonomous navigation visualization.)*

## Installation

This project is built on **ROS2**. To get started, ensure you have a compatible ROS2 environment set up.

### Requirements

  * **Operating System:** Ubuntu 22.04 (Recommended)
  * **ROS2 Distribution:** Humble Hawksbill (or later compatible distribution)
  * **Target Hardware:**
      * NVIDIA Jetson Orin series (e.g., Orin Nano, Orin NX, AGX Orin)
      * Rockchip RK3588 based boards
  * Python 3.10+
  * Specific hardware components (e.g., compatible camera, laser module, lamp, speaker, GPIO-controlled board)

### Steps

1.  **Clone the repository:**

    ```bash
    git clone https://gitlab.com/skyautonet_sw/skyautonet_birdro.git
    cd skyautonet_birdro
    ```

2.  **Initialize and update ROS2 dependencies:**

    ```bash
    rosdep install -i --from-paths src --rosdistro humble -y
    ```

3.  **Build the workspace:**

    ```bash
    colcon build
    ```

4.  **Source the workspace:**

    ```bash
    source install/setup.bash
    ```

    *(You might want to add this line to your `~/.bashrc` for convenience.)*

## Usage

To launch the full **skyautonet\_birdro** system:

1.  **Start the core system:**

    ```bash
    ros2 launch skyautonet_birdro core.launch.py
    ```

    *(This is a placeholder; you'll likely have a main launch file that brings up all necessary nodes. Ensure your launch file name ends with `.py` or `.xml` as per ROS2 conventions.)*

2.  **Monitoring and Control:**

      * **Bird Detection Visualization:** View the camera feed with detected birds in RViz or a dedicated GUI.
      * **Autonomous Navigation Status:** Monitor the robot's localization and path planning.
      * **Deterrent Activation:** Observe laser, lamp, and speaker operation.

    *(Provide specific `ros2 run` or `ros2 launch` commands for key functionalities, e.g., to run a specific bird detection node or control interface.)*

    **Example: Running the Bird Detector (if separated):**

    ```bash
    ros2 run skyautonet_perception bird_detector_node
    ```

## Support

For any questions or issues, please use the GitLab issue tracker:
[https://gitlab.com/skyautonet\_sw/skyautonet\_birdro/-/issues](https://www.google.com/search?q=https://gitlab.com/skyautonet_sw/skyautonet_birdro/-/issues)

## Roadmap

Our future development plans include:

  * **Enhanced AI Models:** Improving bird detection and classification accuracy under various conditions.
  * **Dynamic Deterrence Strategies:** Implementing more adaptive and learning-based deterrence methods.
  * **Multi-robot Coordination:** Exploring capabilities for collaborative operation of multiple birdro units.
  * **Energy Optimization:** Developing power-efficient operational modes for extended deployment.
  * **Advanced Obstacle Avoidance:** Integrating more robust navigation capabilities for complex environments.

## Contributing

This project is developed by the Skyautonet internal team. Contributions are handled internally via our GitLab workflow. Please coordinate with your project lead for any proposed changes or features.

## Authors and Acknowledgment

  * Developed by the **Skyautonet Software Team**.
  * **Senior Researchers:**
      * Sungyooun Jin (sungyooun.jin@skyautonet.com)
      * Kijong Gong (kijong.gong@skyautonet.com)
      * Kyunghwan Kim (kyunghwan.kim@skyautonet.com)
      * Sanghyeok Bae (sanghyeok.bae@skyautonet.com)
      * Junghyun Park (junghyun.park@skyautonet.com)
      * Jihun Kim (jihun.kim@skyautonet.com)
  * **Researchers:**
      * Jiseong Ryu (jiseong.ryu@skyautonet.com)
      * Minsoo Lee (minsoo.lee@skyautonet.com)
      * Damgi Ahn (damgi.ahn@skyautonet.com)
      * Sanghyo Kim (sanghyo.kim@skyautonet.com)
      * Doyoung Lee (doyoung.lee@skyautonet.com)
      * Jaehun Lee (jaehun.lee@skyautonet.com)
  * Special thanks to the open-source community for providing valuable tools and libraries, especially **ROS2** and **Autoware.Universe**.

## License

This project is proprietary to Skyautonet and is not intended for public distribution or use. All rights reserved by Skyautonet.

## Project status

**Active development.** We are continuously working on improving features and performance.