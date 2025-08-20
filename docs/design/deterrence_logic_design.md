# 조류 퇴치 로직 상세 설계 (모드 전환, 퇴치 전략)

## 1. 개요

이 문서는 `skyautonet_birdro` 시스템의 핵심 기능인 **조류 퇴치(Deterrence) 로직**에 대한 상세 설계를 다룹니다. 새 검출 및 추적 결과를 기반으로 로봇이 자율적으로 퇴치 모드를 결정하고, 다양한 퇴치 수단(레이저, 램프, 소리)을 조합하여 새를 효과적으로 퇴치하는 전략을 정의합니다.

## 2. 주요 서브 모듈

`skyautonet_deterrence` 패키지는 다음과 같은 핵심 서브 모듈로 구성됩니다.

* **`skyautonet_deterrence_manager`**:
    * **목표**: 조류 퇴치 프로세스의 **상위 제어 로직**을 담당합니다. 새 검출/추적 정보, 로봇의 현재 상태, 환경 요인 등을 고려하여 최적의 퇴치 모드를 결정하고, 해당 모드에 맞는 퇴치 액션을 하위 컨트롤러에 지시합니다.
    * **상태 머신**:
        * **`Idle Mode`**: 새가 감지되지 않거나, 퇴치 임무가 없을 때.
        * **`Detection Mode`**: 새가 감지되었으나 아직 추적 중이거나 퇴치 행동 전 단계.
        * **`Tracking Mode`**: 특정 새가 추적되고 있으며, 퇴치 대상을 선정하는 단계.
        * **`Deterrence Mode`**: 새를 향해 적극적인 퇴치 행동을 수행하는 단계. (세부 전략: `Laser`, `Lamp`, `Sound`, `Combined`)
        * **`Cooldown Mode`**: 퇴치 후 일정 시간 동안 재퇴치를 방지하는 단계.
    * **입력**: `skyautonet_perception_msgs/msg/TrackedBirds` (추적된 새 정보), `nav_msgs/msg/Odometry` (로봇 위치).
    * **출력**: `skyautonet_deterrence_msgs/msg/DeterrenceCommand` (퇴치 장치 활성화/비활성화 명령), `skyautonet_autonomy_msgs/msg/MissionGoal` (자율주행 미션 변경 요청 - 예: 퇴치 대상 접근).
* **`skyautonet_laser_controller`**:
    * **목표**: `deterrence_manager`의 명령에 따라 **레이저 장치**를 제어합니다.
    * **기술 스택**: GPIO 제어 또는 시리얼 통신을 통한 레이저 모듈 제어.
    * **입력**: `skyautonet_deterrence_msgs/msg/DeterrenceCommand` (레이저 활성화, 방향, 강도 등).
    * **출력**: (없음 또는 상태 피드백).
* **`skyautonet_lamp_controller`**:
    * **목표**: `deterrence_manager`의 명령에 따라 **램프(플래시, 스트로브 등)**를 제어합니다.
    * **기술 스택**: GPIO 제어.
    * **입력**: `skyautonet_deterrence_msgs/msg/DeterrenceCommand` (램프 활성화, 패턴 등).
    * **출력**: (없음 또는 상태 피드백).
* **`skyautonet_speaker_controller`**:
    * **목표**: `deterrence_manager`의 명령에 따라 **스피커**를 통해 다양한 퇴치 음원을 재생합니다.
    * **기술 스택**: 오디오 라이브러리 (예: PortAudio, PyAudio)를 통한 사운드 파일 재생.
    * **입력**: `skyautonet_deterrence_msgs/msg/DeterrenceCommand` (사운드 파일 ID, 볼륨 등).
    * **출력**: (없음 또는 상태 피드백).

## 3. 퇴치 전략

다양한 퇴치 수단을 조합하여 효과를 극대화하는 전략을 수립합니다.

* **단일 수단**: 레이저, 램프, 소리 중 하나만 사용. (예: 저조도 시 레이저, 야간 시 램프, 특정 새 종류에 대한 소리)
* **복합 수단**:
    * **Laser + Sound**: 새를 직접 겨냥하며 특정 소리 재생.
    * **Lamp + Sound**: 강한 섬광과 함께 경고음 재생.
    * **All Combined**: 필요 시 모든 수단을 동원하여 최대 효과 유도.
* **접근 전략**: 퇴치 대상을 향해 자율주행을 통해 특정 거리까지 접근한 후 퇴치 행동 시작.
* **무작위성**: 새가 퇴치 수단에 적응하지 않도록 퇴치 패턴, 소리 종류, 레이저/램프 작동 시간 등에 무작위성을 도입.

## 4. 데이터 흐름

[skyautonet_bird_tracker] --TrackedBirds--> [skyautonet_deterrence_manager]
[skyautonet_autonomy_localization] --RobotPose--> [skyautonet_deterrence_manager]
|
+--DeterrenceCommand--> [skyautonet_laser_controller]
+--DeterrenceCommand--> [skyautonet_lamp_controller]
+--DeterrenceCommand--> [skyautonet_speaker_controller]

## 5. 성능 및 안전 고려사항

* **퇴치 성공률**: 특정 환경 및 새 종류에 대한 목표 퇴치 성공률 설정.
* **안전**: 레이저 사용 시 사람이나 특정 사물에 대한 안전 거리 및 방향 제한 로직 구현.
* **환경 적응**: 주/야간, 날씨 변화에 따른 퇴치 수단 및 강도 조절.
* **에너지 효율**: 퇴치 수단 사용 시간을 최소화하여 배터리 소모 최적화.

