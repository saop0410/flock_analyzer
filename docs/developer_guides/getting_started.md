# 개발 환경 설정 및 프로젝트 시작 가이드

## 1. 개요

이 문서는 `skyautonet_birdro` 프로젝트 개발을 위한 **개발 환경 설정 방법**과 **프로젝트를 시작하는 기본 절차**를 안내합니다.

## 2. 개발 환경 요구사항

* **운영체제**: Ubuntu 22.04 LTS (Jammy Jellyfish)
* **ROS2 배포판**: Humble Hawksbill
* **컴파일러**: GCC 11+
* **Python**: 3.10+
* **Git**: 최신 버전

## 3. ROS2 Humble 설치

아래 공식 문서에 따라 ROS2 Humble을 설치합니다.
[ROS2 Humble 설치 가이드](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html)

**간략한 설치 절차:**

1.  로케일 설정
2.  ROS2 APT 레포지토리 추가
3.  ROS 키 추가
4.  ROS2 Humble 데스크톱 버전 설치
    ```bash
    sudo apt install ros-humble-desktop
    ```
5.  `~/.bashrc`에 ROS2 환경 설정 추가 (필수!)
    ```bash
    echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
    source ~/.bashrc
    ```
6.  `rosdep` 초기화 및 업데이트
    ```bash
    sudo apt install python3-rosdep
    sudo rosdep init # 처음 한 번만 실행
    rosdep update
    ```
7.  `colcon` 설치
    ```bash
    sudo apt install python3-colcon-common-extensions
    ```

## 4. 프로젝트 클론 및 워크스페이스 설정

1.  **프로젝트 클론**:
    ```bash
    git clone [https://gitlab.com/skyautonet_sw/skyautonet_birdro.git](https://gitlab.com/skyautonet_sw/skyautonet_birdro.git)
    cd skyautonet_birdro
    ```
2.  **ROS 워크스페이스로 이동**:
    ```bash
    cd skyautonet_birdro_ws/
    ```
3.  **종속성 설치**:
    ```bash
    rosdep install -i --from-paths src --rosdistro humble -y
    ```
4.  **워크스페이스 빌드**:
    ```bash
    colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
    ```
    * `--symlink-install`: 개발 중 소스 코드 변경 시 재빌드 없이 바로 적용 가능 (Python 노드)
    * `-DCMAKE_BUILD_TYPE=Release`: 최적화된 릴리즈 빌드 (디버깅 시 `Debug`로 변경 가능)
5.  **환경 설정 소싱**:
    ```bash
    source install/setup.bash
    ```
    * 매번 터미널을 열 때마다 이 명령어를 실행해야 합니다. `~/.bashrc`에 추가하는 것을 권장합니다.
        ```bash
        echo "source ~/skyautonet_birdro/skyautonet_birdro_ws/install/setup.bash" >> ~/.bashrc
        source ~/.bashrc
        ```

## 5. 첫 번째 실행 (예시)

워크스페이스 빌드 및 소싱이 완료되면, 기본 시스템을 실행해 볼 수 있습니다.

ros2 launch skyautonet_launch birdro_full_system.launch.py

이 명령어를 실행하면 skyautonet_birdro 시스템의 핵심 ROS2 노드들이 시작됩니다. RViz2 등을 통해 시각화를 확인하며 정상 동작 여부를 검증할 수 있습니다.

## 6. 개발 도구 설정 (VS Code 권장)

  VS Code 설치: Visual Studio Code 공식 웹사이트에서 설치합니다.

  필수 확장:

      C++: ms-vscode.cpptools (Microsoft C/C++ extension)

      Python: ms-python.python (Python extension)

      ROS: ms-iot.vscode-ros (ROS extension for VS Code)

      EditorConfig: EditorConfig.EditorConfig

      clang-format: xaver.clang-format

      flake8: eamodio.gitlens (or other Python linting extensions)

  설정: .editorconfig, .clang-format, .flake8 파일을 활용하여 코드 포매팅 및 스타일을 자동으로 적용하도록 설정합니다.