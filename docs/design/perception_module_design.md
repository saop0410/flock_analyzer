# 인지 모듈 상세 설계 (새 검출, 추적 등)

## 1. 개요

이 문서는 `skyautonet_birdro` 시스템의 **인지(Perception) 모듈**에 대한 상세 설계를 다룹니다. 주요 목표는 카메라 데이터를 활용하여 새를 정확하게 검출하고 추적하며, 자율주행을 위한 일반 객체 인지 기능을 제공하는 것입니다.

## 2. 주요 서브 모듈

`skyautonet_perception` 패키지는 다음과 같은 서브 모듈로 구성됩니다.

* **`skyautonet_bird_detector`**:
    * **목표**: 이미지 스트림에서 새의 위치(바운딩 박스)와 종류를 실시간으로 검출합니다.
    * **기술 스택**: YOLOvX 또는 Custom CNN 모델 (PyTorch/TensorFlow). Jetson Orin의 경우 TensorRT 최적화 고려.
    * **입력**: `sensor_msgs/msg/Image` (카메라 드라이버로부터)
    * **출력**: `skyautonet_perception_msgs/msg/DetectedBirds` (새 바운딩 박스, 클래스, 신뢰도)
* **`skyautonet_bird_tracker`**:
    * **목표**: 검출된 새들의 ID를 유지하며 시퀀스에 걸쳐 궤적을 추적합니다.
    * **기술 스택**: Kalman Filter, DeepSORT 또는 ByteTrack 등.
    * **입력**: `skyautonet_perception_msgs/msg/DetectedBirds` (새 검출기로부터)
    * **출력**: `skyautonet_perception_msgs/msg/TrackedBirds` (새 ID, 궤적, 상태)
* **`skyautonet_object_detection`**:
    * **목표**: 로봇 주변의 일반적인 장애물(사람, 차량 등)을 검출합니다.
    * **기술 스택**: 표준 객체 검출 모델 (COCO 데이터셋 기반).
    * **입력**: `sensor_msgs/msg/Image`
    * **출력**: `vision_msgs/msg/Detection2DArray` 또는 유사 메시지 (장애물 바운딩 박스)
* **`skyautonet_sensor_processing`**:
    * **목표**: 카메라 이미지 보정, 왜곡 보정 등 센서 데이터 전처리를 수행합니다.
    * **기술 스택**: OpenCV, ROS2 Image Pipeline.
    * **입력**: RAW `sensor_msgs/msg/Image`
    * **출력**: Rectified `sensor_msgs/msg/Image`

## 3. 데이터 흐름

[skyautonet_camera_driver] --Image--> [skyautonet_sensor_processing] --Rectified Image--> [skyautonet_bird_detector] --DetectedBirds--> [skyautonet_bird_tracker] --TrackedBirds--> [skyautonet_deterrence_manager]
|
+---Rectified Image--> [skyautonet_object_detection] --Detection2DArray--> [skyautonet_autonomy_planning]

## 4. 성능 요구사항

* **새 검출**: 최소 10 FPS @ 720p (최소 90% 정확도)
* **새 추적**: 최소 10 FPS (낮은 ID 스위치 비율)
* **일반 객체 검출**: 최소 5 FPS @ 720p

## 5. 학습 및 배포 전략

* **데이터셋**: 실제 환경에서 수집된 새 이미지 데이터셋을 구축하고 레이블링합니다.
* **학습 환경**: GPU 서버 (RTX 4090 이상)에서 모델을 학습합니다.
* **배포**: Jetson Orin 및 RK3588 보드에 최적화된 모델 (예: TensorRT, ONNX Runtime)을 배포합니다.
