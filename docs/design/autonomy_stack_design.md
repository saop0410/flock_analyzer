# 자율주행 스택 상세 설계 (현지화, 플래닝, 제어 등)

## 1. 개요

이 문서는 `skyautonet_birdro` 로봇의 **자율주행(Autonomy) 스택**에 대한 상세 설계를 정의합니다. 로봇이 주어진 환경에서 스스로 이동하고, 경로를 계획하며, 장애물을 회피하고 목표 지점까지 도달하는 기능을 구현하는 것이 목표입니다.

## 2. 주요 서브 모듈

`skyautonet_autonomy` 패키지는 다음과 같은 핵심 서브 모듈로 구성됩니다.

* **`skyautonet_localization`**:
    * **목표**: 로봇의 현재 위치(x, y, yaw)를 지도 상에서 정확하게 추정합니다.
    * **기술 스택**:
        * **초기 버전**: AMCL (Adaptive Monte Carlo Localization) with LiDAR/Map.
        * **고도화**: Visual Odometry, LiDAR Odometry, GPS/IMU 융합.
    * **입력**: `sensor_msgs/msg/LaserScan` (LiDAR), `sensor_msgs/msg/Imu`, `nav_msgs/msg/Odometry` (바퀴 오도메트리).
    * **출력**: `nav_msgs/msg/Odometry` (로봇의 글로벌 자세).
* **`skyautonet_planning`**:
    * **목표**: 로봇의 현재 위치와 목표 지점(또는 순찰 경로)을 기반으로 안전하고 효율적인 경로를 생성합니다.
    * **기술 스택**:
        * **글로벌 플래닝**: Dijkstra, A* (맵 기반).
        * **로컬 플래닝**: DWA (Dynamic Window Approach), TEB Local Planner (장애물 회피).
    * **입력**: `nav_msgs/msg/Odometry` (현재 자세), `nav_msgs/msg/Path` (글로벌 목표 경로), `sensor_msgs/msg/PointCloud2` 또는 `sensor_msgs/msg/LaserScan` (로컬 장애물 정보).
    * **출력**: `geometry_msgs/msg/Twist` (로컬 계획된 속도 명령) 또는 `nav_msgs/msg/Path` (로컬 계획된 경로).
* **`skyautonet_control`**:
    * **목표**: 플래닝 모듈에서 생성된 속도 명령 또는 경로를 추종하도록 로봇의 구동계를 제어합니다.
    * **기술 스택**: PID 컨트롤러, MPC (Model Predictive Control) 등.
    * **입력**: `geometry_msgs/msg/Twist` (명령 속도), `nav_msgs/msg/Odometry` (현재 자세).
    * **출력**: `skyautonet_drivetrain_msgs/msg/MotorCommand` (하위 구동계 드라이버로 전달될 구체적인 모터 제어 명령).
* **`skyautonet_vehicle_interface`**:
    * **목표**: `skyautonet_control`에서 생성된 추상적인 제어 명령(`Twist` 또는 `MotorCommand`)을 실제 `skyautonet_drivetrain_driver`가 이해할 수 있는 저수준 명령으로 변환합니다.
    * **기술 스택**: ROS2 Node (Python/C++).
    * **입력**: `skyautonet_drivetrain_msgs/msg/MotorCommand`.
    * **출력**: `skyautonet_drivetrain_msgs/msg/RawMotorPWM` (예: 각 모터의 PWM 값)

## 3. 데이터 흐름

[센서 드라이버] --LiDAR/Odometry--> [skyautonet_localization] --Pose--> [skyautonet_planning] --Velocity Command--> [skyautonet_control] --MotorCommand--> [skyautonet_vehicle_interface] --RawMotorPWM--> [skyautonet_drivetrain_driver]
^
|
[skyautonet_object_detection] (장애물 정보)


## 4. 지도 및 환경 모델링

* **정적 맵**: 사전에 구축된 환경 맵 (Occupancy Grid Map 또는 Point Cloud Map)을 사용하여 현지화 및 글로벌 플래닝에 활용합니다.
* **동적 맵**: `skyautonet_object_detection`에서 얻은 정보를 기반으로 로컬 환경의 동적 장애물을 모델링합니다.

## 5. 성능 요구사항

* **현지화 정확도**: 특정 환경에서 맵 대비 오차 10cm 이내.
* **경로 추종**: 목표 경로에서 벗어나는 오차 5cm 이내.
* **장애물 회피**: 실시간으로 장애물을 감지하고 안전하게 회피.