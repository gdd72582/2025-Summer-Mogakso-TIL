# 2주차: YOLO + 자율주행 인지 실험

## 📋 실험 개요

### 목표
표지판/보행자/차량 탐지 모델을 학습하여 자율주행 의사결정에 쓸 수 있는 신뢰도 있는 탐지 결과 제공

### 성공 기준
- **검증 성능**: mAP@0.5 ≥ 0.60, mAP@0.5:0.95 ≥ 0.35
- **실시간성**: ≥ 20 FPS (640 입력, 단일 GPU 기준), 추론 지연 ≤ 50 ms
- **통합**: ROS 토픽(`/detections`)로 변환하여 장애물 회피 데모 성공

---

## 🗂️ 프로젝트 구조

```
2주차/
├── README.md                    # 이 파일
├── requirements.txt             # 필요한 패키지 목록
├── config/
│   ├── data.yaml               # 데이터셋 설정
│   └── experiment_config.yaml  # 실험 설정
├── data/
│   └── datasets/               # 데이터셋 저장소
├── scripts/
│   ├── download_dataset.py     # 데이터셋 다운로드
│   ├── train_model.py          # 모델 학습
│   └── benchmark_model.py      # 성능 벤치마크
└── experiments/
    └── results_summary.md      # 실험 결과 요약
```

---

## 🚀 빠른 시작

### 1. 환경 설정
```bash
# 가상환경 생성 및 활성화
python -m venv yolo_env
source yolo_env/bin/activate  # Windows: yolo_env\Scripts\activate

# 패키지 설치
pip install -r requirements.txt
```

### 2. 데이터셋 준비
```bash
# 데이터셋 다운로드 (샘플 데이터셋 생성)
python scripts/download_dataset.py --sample

# 실제 데이터셋 다운로드 (선택사항)
python scripts/download_dataset.py --kitti
python scripts/download_dataset.py --bdd100k
```

### 3. 베이스라인 실험
```bash
# YOLOv8s 베이스라인 학습
python scripts/train_model.py --config config/experiment_config.yaml --exp_name exp0_baseline
```

### 4. 실험 실행
```bash
# 해상도 실험
python scripts/train_model.py --config config/experiment_config.yaml --exp_name exp1_resolution --imgsz 800

# 데이터 증강 실험
python scripts/train_model.py --config config/experiment_config.yaml --exp_name exp2_augmentation --mosaic 0.0

# Ablation 실험 (설정 파일의 ablation_studies 섹션 사용)
python scripts/train_model.py --config config/experiment_config.yaml --ablation

# 모델 검증
python scripts/train_model.py --config config/experiment_config.yaml --validate path/to/model.pt
```

---

## 📊 실험 계획

### Day 1: 환경 세팅 및 데이터 준비
- [ ] 환경 설정 및 패키지 설치
- [ ] 데이터셋 다운로드 (KITTI/BDD100K subset)
- [ ] 데이터 전처리 및 라벨 포맷 통일
- [ ] 베이스라인 노트북 작성

### Day 2: 베이스라인 학습
- [ ] YOLOv8n/s 모델 학습 (100 epochs)
- [ ] 기본 성능 평가
- [ ] 결과 시각화

### Day 3: 해상도 실험 (Ablation #1)
- [ ] 640 vs 800 해상도 비교
- [ ] Batch size 16 vs 32 비교
- [ ] 정규화 기법 비교

### Day 4: 데이터 증강 실험 (Ablation #2)
- [ ] Mosaic, HSV, Mixup 조합 실험
- [ ] NMS IoU 임계값 조정
- [ ] 작은 객체 탐지 개선

### Day 5: 최적화 실험
- [ ] 모델 경량화 (v8n vs v8s)
- [ ] Loss function 최적화
- [ ] 하이퍼파라미터 튜닝

### Day 6: 모델 내보내기 및 벤치마크
- [ ] ONNX/TensorRT 변환
- [ ] FPS 및 지연시간 측정
- [ ] 성능 최적화

### Day 7: 결과 분석 및 문서화
- [ ] 실험 결과 종합 분석
- [ ] 최적 모델 선택
- [ ] 최종 리포트 작성

---

## 🔧 주요 설정

### 데이터셋 설정 (config/data.yaml)
```yaml
# 데이터셋 경로 및 클래스 정보
path: ./data/datasets/kitti_subset
train: images/train
val: images/val
test: images/test

# 클래스 정보
nc: 4  # 클래스 수
names: ['car', 'person', 'traffic_light', 'stop_sign']

# 클래스별 정보
class_info:
  car:
    description: "Vehicle (car, truck, bus)"
    color: [255, 0, 0]  # Red
    priority: 1
  person:
    description: "Pedestrian"
    color: [0, 255, 0]  # Green
    priority: 2
  traffic_light:
    description: "Traffic light"
    color: [0, 0, 255]  # Blue
    priority: 3
  stop_sign:
    description: "Stop sign"
    color: [255, 255, 0]  # Yellow
    priority: 4
```

### 실험 설정 (config/experiment_config.yaml)
```yaml
# 모델 설정
model:
  name: "yolov8s"
  pretrained: true
  weights: "yolov8s.pt"

# 학습 설정
training:
  epochs: 100
  batch_size: 16
  imgsz: 640
  lr0: 0.01
  lrf: 0.1
  momentum: 0.937
  weight_decay: 0.0005

# 데이터 증강 설정
augmentation:
  hsv_h: 0.015
  hsv_s: 0.7
  hsv_v: 0.4
  degrees: 0.0
  translate: 0.1
  scale: 0.5
  flipud: 0.0
  fliplr: 0.5
  perspective: 0.0
  mosaic: 1.0
  mixup: 0.0

# 추론 설정
inference:
  conf_thres: 0.25
  iou_thres: 0.6
  max_det: 300

# Ablation 실험 설정
ablation_studies:
  resolution_experiments:
    - name: "baseline_640"
      imgsz: 640
    - name: "high_res_800"
      imgsz: 800
    - name: "low_res_512"
      imgsz: 512
```

---

## 📈 성능 평가

### 평가 지표
- **mAP@0.5**: IoU=0.5에서의 평균 정확도
- **mAP@0.5:0.95**: IoU=0.5~0.95에서의 평균 정확도
- **FPS**: 초당 처리 프레임 수
- **Latency**: 추론 지연시간 (ms)
- **GPU Memory**: GPU 메모리 사용량

### 실험 결과 표
| 실험 | 변경점 | mAP@0.5 | mAP@0.5:0.95 | FPS | Latency(ms) | GPU Mem(MB) | 주석 |
|------|--------|---------|--------------|-----|-------------|-------------|------|
| Baseline | YOLOv8s, 640 | - | - | - | - | - | 기본 설정 |
| Resolution | 800 | - | - | - | - | - | 해상도 증가 |
| Resolution | 512 | - | - | - | - | - | 해상도 감소 |
| No Mosaic | mosaic=0.0 | - | - | - | - | - | 모자이크 비활성화 |
| Strong Aug | 강화된 증강 | - | - | - | - | - | 강화된 데이터 증강 |
| Lightweight | YOLOv8n | - | - | - | - | - | 경량 모델 |
| Large Batch | batch=32 | - | - | - | - | - | 배치 크기 증가 |

---

## 🛠️ 유용한 명령어

### 학습
```bash
# 기본 학습
yolo train data=config/data.yaml model=yolov8s.pt imgsz=640 batch=16 epochs=100

# 커스텀 설정으로 학습
python scripts/train_model.py --config config/experiment_config.yaml --exp_name custom_exp
```

### 평가
```bash
# 모델 평가
yolo val model=runs/train/exp/weights/best.pt data=config/data.yaml

# 성능 벤치마크
python scripts/benchmark_model.py --model best.pt --img 640 --warmup 500 --iters 2000

# 다양한 이미지 크기에서 벤치마크
python scripts/benchmark_model.py --model best.pt --size_benchmark --sizes 320 640 800 1024

# 결과 저장
python scripts/benchmark_model.py --model best.pt --output benchmark_results.json --report benchmark_report.md
```

### 내보내기
```bash
# ONNX 변환
yolo export model=best.pt format=onnx

# TensorRT 변환
yolo export model=best.pt format=engine device=0
```

---

## 📚 참고 자료

### 공식 문서
- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [ONNX Documentation](https://onnx.ai/)

### 데이터셋
- [KITTI Dataset](http://www.cvlibs.net/datasets/kitti/)
- [BDD100K Dataset](https://bdd100k.com/)
- [COCO Dataset](https://cocodataset.org/)

### 논문
- [YOLOv8 Paper](https://arxiv.org/abs/2304.00501)
- [YOLO Evolution](https://arxiv.org/abs/2209.02976)

### 스크립트 사용법
- **download_dataset.py**: 데이터셋 다운로드 및 샘플 데이터셋 생성
- **train_model.py**: YOLO 모델 학습, 검증, Ablation 실험
- **benchmark_model.py**: 모델 성능 벤치마크 (FPS, 지연시간, 메모리)

---

## 🤝 기여 가이드

### 코드 스타일
- PEP 8 준수
- 함수 및 클래스에 docstring 작성
- 변수명은 명확하고 의미있게 작성

### 실험 기록
- 모든 실험은 `experiments/` 폴더에 저장
- 설정 파일과 결과를 함께 보관
- 실험 결과는 `experiments/results_summary.md`에 기록

### 이슈 리포트
- 버그 발견 시 즉시 이슈 등록
- 재현 가능한 최소 예제 포함
- 환경 정보 (OS, Python 버전, GPU 등) 명시

### 파일 구조
- **config/**: 설정 파일들 (data.yaml, experiment_config.yaml)
- **scripts/**: 실행 스크립트들 (download_dataset.py, train_model.py, benchmark_model.py)
- **experiments/**: 실험 결과 및 요약 (results_summary.md)
- **data/**: 데이터셋 저장소

---

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

---

*마지막 업데이트: 2024년 1월* 