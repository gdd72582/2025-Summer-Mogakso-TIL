# 2주차: Object Detection 및 YOLO 구조 학습

## Object Detection 개요

### 1. Object Detection이란?
Object Detection은 컴퓨터 비전의 핵심 기술로, 이미지나 영상에서 특정 객체의 위치와 종류를 동시에 찾아내는 기술입니다.

**주요 특징:**
- **Classification + Localization:** 객체 분류와 위치 탐지를 동시 수행
- **Bounding Box:** 객체를 감싸는 사각형 좌표 (x, y, width, height)
- **Multi-class:** 여러 종류의 객체를 동시에 탐지 가능
- **Real-time:** 실시간 처리 가능한 알고리즘들 존재

---

### 2. Object Detection의 발전 과정

#### 2.1 전통적 방법 (Traditional Methods)
- **HOG (Histogram of Oriented Gradients):** 경계선 방향 히스토그램
- **SIFT (Scale-Invariant Feature Transform):** 스케일 불변 특징점
- **SURF (Speeded Up Robust Features):** SIFT의 고속 버전
- **Haar Cascade:** 얼굴 검출 등에 사용되는 캐스케이드 분류기

#### 2.2 딥러닝 기반 방법
- **R-CNN (2014):** Region-based CNN, 슬라이딩 윈도우 방식
- **Fast R-CNN (2015):** R-CNN의 속도 개선 버전
- **Faster R-CNN (2016):** Region Proposal Network 도입
- **YOLO (2016):** One-stage detector, 실시간 처리 가능
- **SSD (2016):** Multi-scale feature maps 활용
- **RetinaNet (2017):** Focal Loss로 정확도 향상

---

### 3. 주요 데이터셋

#### 3.1 COCO (Common Objects in Context)
**특징:**
- 330K 이미지, 2.5M 인스턴스
- 80개 클래스 (person, car, dog, cat 등)
- 다양한 크기와 복잡한 장면
- Instance segmentation, keypoint detection 포함

**평가 지표:**
- mAP (mean Average Precision)
- AP@0.5, AP@0.75
- AP_small, AP_medium, AP_large

#### 3.2 PASCAL VOC
**특징:**
- 20개 클래스
- 11K 이미지
- Object Detection의 표준 벤치마크
- 2005-2012년까지 매년 개최

#### 3.3 자율주행 관련 데이터셋
**KITTI:**
- 자동차, 보행자, 자전거 탐지
- 3D 바운딩 박스
- 스테레오 카메라 + LiDAR

**BDD100K:**
- 100K 이미지
- 자율주행 시나리오
- 날씨, 시간대별 다양성

**nuScenes:**
- 3D Object Detection
- 6개 카메라 + LiDAR
- 23개 클래스

---

## YOLO (You Only Look Once) 구조

### 4. YOLO의 핵심 개념

#### 4.1 One-stage Detector
- **전통적 방법:** Region Proposal → Classification → Bounding Box Regression
- **YOLO 방식:** 한 번의 네트워크 통과로 모든 작업 수행
- **장점:** 빠른 속도, 실시간 처리 가능
- **단점:** 작은 객체 탐지 정확도 상대적으로 낮음

#### 4.2 Grid-based Detection
- 입력 이미지를 S×S 그리드로 분할
- 각 그리드 셀은 객체의 중심점을 포함하는지 여부 판단
- 각 셀에서 B개의 바운딩 박스 예측

---

### 5. YOLO 버전별 발전 과정

#### 5.1 YOLO v1 (2016)
**구조:**
- 24개 Convolutional layers + 2개 Fully connected layers
- GoogLeNet 기반 backbone
- 7×7 그리드, 2개 바운딩 박스 per cell

**특징:**
- 실시간 처리 (45 FPS)
- 글로벌 컨텍스트 고려
- 작은 객체 탐지 어려움

#### 5.2 YOLO v2 (2017)
**개선사항:**
- **Batch Normalization:** 정규화로 성능 향상
- **High Resolution Classifier:** 448×448 해상도
- **Convolutional with Anchor Boxes:** 앵커 박스 도입
- **Dimension Clusters:** K-means로 앵커 박스 크기 최적화
- **Direct location prediction:** 위치 예측 개선
- **Fine-Grained Features:** 13×13 + 26×26 feature maps

#### 5.3 YOLO v3 (2018)
**주요 개선:**
- **Multi-scale Prediction:** 3개 스케일 (13×13, 26×26, 52×52)
- **Better Backbone:** Darknet-53 (ResNet 스타일)
- **Feature Pyramid Network (FPN):** 다양한 크기 객체 탐지
- **Binary Cross-entropy:** 클래스 예측 개선
- **Better Loss Function:** Focal Loss 적용

#### 5.4 YOLO v4 (2020)
**기술적 혁신:**
- **CSPDarknet53:** Cross Stage Partial connections
- **PANet:** Path Aggregation Network
- **Mosaic Data Augmentation:** 4개 이미지 조합
- **Self-Adversarial Training:** 데이터 증강 기법
- **CmBN:** Cross mini-Batch Normalization

#### 5.5 YOLO v5 (2020)
**실용적 개선:**
- **PyTorch 기반:** 더 쉬운 구현과 배포
- **Auto-anchor:** 자동 앵커 박스 생성
- **Mixed Precision:** 메모리 효율성
- **Model Export:** 다양한 플랫폼 지원

#### 5.6 YOLO v6, v7, v8
**최신 발전:**
- **YOLO v6:** Edge device 최적화
- **YOLO v7:** Real-time object detection 최적화
- **YOLO v8:** Instance segmentation, pose estimation 추가

---

### 6. YOLO의 핵심 구성 요소

#### 6.1 Backbone Network
**역할:** 특징 추출
- **Darknet:** YOLO 전용 CNN 아키텍처
- **ResNet, EfficientNet:** 다른 백본 네트워크 활용 가능
- **Depth-wise Separable Convolution:** 효율성 향상

#### 6.2 Neck Network
**역할:** 특징 융합 및 스케일 조정
- **FPN (Feature Pyramid Network):** 다양한 스케일 특징 융합
- **PANet:** 하향식 + 상향식 특징 전파
- **BiFPN:** 양방향 특징 피라미드 네트워크

#### 6.3 Head Network
**역할:** 최종 예측
- **Classification Head:** 객체 클래스 예측
- **Regression Head:** 바운딩 박스 좌표 예측
- **Objectness Head:** 객체 존재 여부 예측

---

### 7. YOLO의 Loss Function

#### 7.1 구성 요소
**Total Loss = Classification Loss + Localization Loss + Confidence Loss**

#### 7.2 각 Loss의 역할
**Classification Loss:**
- 객체의 클래스 예측 정확도
- Cross-entropy loss 사용

**Localization Loss:**
- 바운딩 박스 좌표 예측 정확도
- MSE (Mean Squared Error) 또는 IoU-based loss

**Confidence Loss:**
- 객체 존재 여부 예측 정확도
- Binary cross-entropy loss

#### 7.3 IoU (Intersection over Union)
- 바운딩 박스 예측 정확도 측정
- Ground truth와 예측 박스의 겹침 정도
- IoU = Intersection Area / Union Area

---

### 8. YOLO의 장단점

#### 8.1 장점
- **실시간 처리:** 빠른 추론 속도
- **글로벌 컨텍스트:** 전체 이미지 정보 활용
- **단순한 구조:** 이해하기 쉬운 아키텍처
- **다양한 응용:** 다양한 도메인에 적용 가능

#### 8.2 단점
- **작은 객체 탐지:** 상대적으로 낮은 정확도
- **밀집 객체:** 겹치는 객체 탐지 어려움
- **정확도 vs 속도:** 트레이드오프 존재
- **데이터 의존성:** 대량의 라벨링된 데이터 필요

---

### 9. YOLO의 실제 응용

#### 9.1 자율주행
- **차량, 보행자, 신호등 탐지**
- **실시간 처리로 안전성 확보**
- **다양한 환경 조건 대응**

#### 9.2 보안 및 감시
- **침입자 탐지**
- **이상 행동 감지**
- **CCTV 분석**

#### 9.3 의료 영상
- **종양 탐지**
- **기관 인식**
- **의료 기기 위치 파악**

#### 9.4 산업 자동화
- **품질 검사**
- **로봇 조작**
- **재고 관리**

---

### 10. 성능 평가 지표

#### 10.1 정확도 관련
- **Precision:** 정확한 탐지 비율
- **Recall:** 실제 객체 탐지 비율
- **F1-Score:** Precision과 Recall의 조화평균
- **mAP:** 여러 IoU 임계값에서의 평균 정확도

#### 10.2 속도 관련
- **FPS (Frames Per Second):** 초당 처리 프레임 수
- **Inference Time:** 추론 시간
- **Throughput:** 처리량

#### 10.3 메모리 관련
- **Model Size:** 모델 파라미터 수
- **Memory Usage:** 메모리 사용량
- **FLOPs:** 연산량

---

### 11. 최적화 기법

#### 11.1 모델 경량화
- **Pruning:** 불필요한 뉴런 제거
- **Quantization:** 정밀도 낮춤 (FP32 → INT8)
- **Knowledge Distillation:** 큰 모델에서 작은 모델로 지식 전달

#### 11.2 추론 최적화
- **TensorRT:** NVIDIA GPU 최적화
- **ONNX:** 프레임워크 간 호환성
- **TensorFlow Lite:** 모바일/엣지 디바이스 최적화

---

### 12. 실습 및 구현

#### 12.1 환경 설정
- **PyTorch 또는 TensorFlow 설치**
- **YOLO 구현체 선택 (YOLOv5, YOLOv8 등)**
- **GPU 환경 구성 (CUDA, cuDNN)**

#### 12.2 데이터 준비
- **데이터셋 다운로드 (COCO, PASCAL VOC 등)**
- **데이터 전처리 및 증강**
- **라벨링 형식 변환**

#### 12.3 모델 학습
- **하이퍼파라미터 설정**
- **Loss function 구현**
- **학습률 스케줄링**

#### 12.4 평가 및 테스트
- **성능 지표 계산**
- **시각화 및 분석**
- **실제 이미지 테스트**

---

### 13. 향후 발전 방향

#### 13.1 기술적 발전
- **Transformer 기반 아키텍처**
- **Self-supervised Learning**
- **Few-shot Learning**
- **Multi-modal Fusion**

#### 13.2 응용 분야 확장
- **3D Object Detection**
- **Video Object Detection**
- **Instance Segmentation**
- **Pose Estimation**

---

### 14. 학습 리소스

#### 14.1 공식 문서
- **YOLO 공식 GitHub**
- **Darknet 프레임워크**
- **Ultralytics YOLOv8**

#### 14.2 튜토리얼 및 강의
- **Joseph Redmon의 YOLO 논문**
- **CS231n (Stanford)**
- **PyTorch 공식 튜토리얼**

#### 14.3 실습 자료
- **Google Colab 노트북**
- **Kaggle 커널**
- **GitHub 예제 코드**

---

---
*이 문서는 2주차 Object Detection 및 YOLO 구조 학습 내용을 정리한 문서입니다.* 