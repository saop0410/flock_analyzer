# flock_analyzer

이 디렉토리는 FODRO 조류 퇴치 프로젝트의 핵심 알고리즘(새떼 분석, 형태 분류, 방향/밀도/군집 수 계산)을 ROS 환경과 독립적으로 개발하고 테스트하기 위한 공간입니다.

## 1. 환경 설정

먼저 필요한 Python 라이브러리를 설치합니다.

```bash
cd /home/saop/flock_analyzer
pip install -r requirements.txt
```

## 2. 모델 및 데이터 준비

`flock_analyzer` 디렉토리 내에 다음 구조로 모델 파일과 테스트 이미지를 준비해야 합니다.

```
/home/saop/flock_analyzer/
├── models/
│   ├── yolov5_best.pt       # YOLOv5 모델 파일 (새 탐지용)
│   └── analyzer_CNN.pth     # 새떼 형태 분류 CNN 모델 파일
├── dummy_image/
│   └── test_bird.jpg        # 테스트할 이미지 파일
└── train_flock_analyzer/
    └── dataset/             # CNN 모델 학습용 데이터셋
        ├── V-shape/
        ├── clustered/
        ├── scattered/
        ├── linear/
        └── ...
```

*   **`yolov5_best.pt`**: 이 파일은 YOLOv5 모델의 가중치 파일입니다. 현재 프로젝트에는 포함되긴 했지만 성능이 떨어지므로, 직접 학습하거나 외부에서 구하여 이 경로에 넣어주셔야 합니다.
*   **`analyzer_CNN.pth`**: 이 파일은 새떼 형태 분류 CNN 모델의 가중치 파일입니다. `train_flock_analyzer/train_flock_analyzer.py` 스크립트를 사용하여 학습한 후 생성됩니다.
*   **`train_flock_analyzer/dataset/`**: `train_flock_analyzer.py`를 사용하여 CNN 모델을 학습하기 위한 데이터셋 디렉토리입니다. 각 형태별로 폴더를 만들고, 해당 형태의 새떼 이미지(64x64 흑백)를 넣어주세요. `dummy_generator_for_CNN_input.py`를 사용하여 더미 데이터셋을 생성할 수 있습니다.

## 3. 새떼 분석 및 시각화 (`flock_analyzer_inference.py`)

`flock_analyzer_inference.py` 스크립트는 주어진 이미지 파일에 대해 새 탐지 및 새떼 분석을 수행하고, 결과를 시각화하여 이미지 파일로 저장합니다.

```bash
cd /home/saop/flock_analyzer
python flock_analyzer_inference.py --image_dir dummy_image --output_dir visualization/result
```

*   스크립트 실행 전에 `models/` 디렉토리에 `yolov5_best.pt`와 `analyzer_CNN.pth` 파일이 있어야 합니다.
*   `--image_dir` 인수를 통해 분석할 이미지가 있는 디렉토리를 지정합니다.
*   분석 결과는 `--output_dir`로 지정된 디렉토리에 저장됩니다.

## 4. 형태 분류 CNN 모델 학습 (`train_flock_analyzer.py`)

`train_flock_analyzer/train_flock_analyzer.py` 스크립트는 새떼 형태 분류 CNN 모델을 학습하는 데 사용됩니다.

```bash
cd /home/saop/flock_analyzer/train_flock_analyzer
python train_flock_analyzer.py
```

*   **데이터셋 준비:** `train_flock_analyzer/dataset/` 디렉토리 아래에 클래스별 폴더를 만들고, 각 폴더 안에 해당 형태의 새떼 이미지(64x64 흑백)를 넣어주세요. `dummy_generator_for_CNN_input.py`를 실행하여 더미 데이터셋을 생성할 수 있습니다.
*   학습이 완료되면 `models/analyzer_CNN.pth` 파일이 생성됩니다.

## 5. 추가 정보

*   `flock_analyzer_inference.py`와 `train_flock_analyzer.py`는 `flock_analyzer` 패키지의 핵심 로직을 ROS 의존성 없이 재구성한 것입니다.
*   YOLOv5 모델 학습은 이 환경에서 직접 제공되지 않습니다. [Ultralytics YOLOv5 GitHub 저장소](https://github.com/ultralytics/yolov5)를 참조하여 별도로 학습을 진행해야 합니다.