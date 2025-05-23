# Project-yolodetect

YOLOv5를 활용한 얼굴 주요 부위 탐지 AI 모델 개발

## 트레이닝 방법
데이터셋을 만든 후 labelme에서 원하는 부분 라벨링 후, convert.py로 txt형식으로 변환 이때, 라벨링한 순서대로 classes명 변경 후 변환

split_dataset.py로 데이터셋 경로, 저장할 디렉터리 경로 설정 후 실행하여 yolo 트레이닝 형식에 맞게 분리

train.py로 트레이닝 시작, 이때, data.yaml에서 경로를 split_dataset.py로 저장한 경로로 images,val 설정, name도 라벨링된 순서대로 변경 후 실행

test.py로 캠 테스트, weights 경로 원하는 모델의 경로로 지정

## 프로젝트 개요
본 프로젝트는 YOLOv5 모델을 활용하여 얼굴 이미지에서 귀, 콧구멍, 입술과 같은 주요 부위를 탐지하는 딥러닝 모델을 개발한 사례입니다. 개인 얼굴 이미지를 기반으로 직접 라벨링하고 데이터셋을 구성하였으며, YOLO 모델 학습에 최적화된 포맷으로 변환하여 훈련을 진행하였습니다. 최종적으로 70% 이상의 정확도를 달성한 모델을 완성하였습니다.

## 🛠️ 주요 기술 및 도구

<p align="left"> <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/> <img src="https://img.shields.io/badge/YOLOv5-0A0A0A?style=for-the-badge&logo=github&logoColor=white" alt="YOLOv5"/> <img src="https://img.shields.io/badge/Labelme-FF6F00?style=for-the-badge&logo=JSON&logoColor=white" alt="Labelme"/> <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV"/> <img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter"/> </p>

* YOLOv5 (yolov5n.pt): 경량 모델을 사용하여 빠른 학습과 추론 수행

* Labelme: 이미지 라벨링 툴을 활용하여 귀, 콧구멍, 입술 위치 수작업 라벨링

* Python: 데이터 전처리 및 포맷 변환 자동화

* Custom Dataset 구성: YOLO 형식에 맞춰 images/, labels/ 폴더 구조 구성

* OpenCV: 이미지 처리 및 시각화 테스트

## 🧪 개발 과정

### 1. 얼굴 이미지 수집 및 라벨링
직접 촬영한 얼굴 사진을 기반으로 Labelme 툴을 사용하여 귀, 콧구멍, 입술을 폴리곤 형태로 라벨링하고 .json 파일로 저장

### 2. YOLO 형식으로 변환
.json 파일을 Python 스크립트를 통해 .txt 파일로 변환 (YOLO 포맷: 클래스 번호, 중심 좌표, 너비/높이)

### 3. 데이터 구성
images/와 labels/ 폴더에 각각 이미지와 라벨 텍스트 파일을 정리하여 YOLO 학습 형식으로 구성

### 4. 모델 학습
yolov5n.pt (YOLOv5 Nano) 사전학습 모델을 사용하여 훈련
이미지 수가 적은 상황에서 빠른 학습과 일반화 성능 확보에 유리하도록 경량 모델 선택

## 🧩 트러블슈팅 경험

* 문제 상황: 초기 라벨링 파일에서 클래스 네임의 순서가 일관되지 않아, 학습 시 클래스 혼동 발생

* 해결 방법: JSON → YOLO 변환 시 라벨링 순서를 항상 '귀, 콧구멍, 입술'의 고정 순서로 매핑되도록 스크립트를 수정하여 재처리 후 재학습

* 성과: 클래스 간 혼동 없이 안정적인 학습 진행이 가능해졌으며, 정확도 70% 이상을 달성

## 📈 결과 및 성과

* 학습 정확도 70% 이상의 YOLOv5n 기반 얼굴 부위 탐지 모델 완성

* 소량의 데이터와 가벼운 모델 구성으로도 실시간 탐지 가능한 수준의 정확도 확보

* 데이터 수집, 라벨링, 포맷 변환, 모델 학습 등 전체 과정을 직접 수행

![PR_curve](https://github.com/user-attachments/assets/f35e9599-4299-4629-9e43-3af71579a861)


![train_batch0](https://github.com/user-attachments/assets/9df43752-e6ae-4af6-b9b4-57d62d511213)


![val_batch0_labels](https://github.com/user-attachments/assets/1acc7fca-1174-4600-a7ea-2e53dc218267)



## 🧠 배운 점

* 라벨링 순서의 일관성이 학습 정확도에 직접적인 영향을 미친다는 점을 경험적으로 학습

* YOLO 포맷 데이터 구성이 처음에는 복잡할 수 있지만, 스크립트를 통한 자동화로 오류를 줄일 수 있음

* 경량 모델도 적절한 라벨링과 학습 전략을 통해 충분한 성능을 낼 수 있음


