# 5주차 - 멀티모달 AI 자율주행 실험

## 📚 개요

5주차에서는 **멀티모달 AI의 핵심 개념**을 학습하고 이를 **자율주행 시스템에 적용하는 실험**을 진행합니다. CLIP, BLIP-2, BEV 등의 최신 멀티모달 기술을 이해하고, 실제 코드를 통해 자율주행 환경에서의 활용 가능성을 검증합니다.

## 🎯 학습 목표

1. **멀티모달 핵심 개념** 이해 (CLIP/BLIP-2/BEV)
2. **자율주행 데이터셋** 구성 감 잡기 (nuScenes/Waymo)
3. **CLIP 기반 zero-shot/검색·리트리벌** 미니실험 수행
4. **"자율주행에 멀티모달을 어떻게 꽂을지"** 적용 아이디어 메모

## 📁 파일 구조

```
4주차/
├── README.md                           # 현재 파일
├── requirements.txt                    # 필수 패키지 목록
├── clip_experiment.py                  # CLIP 실험 코드
├── blip2_demo.py                       # BLIP-2 데모 코드
└── autonomous_driving_multimodal.py    # 통합 멀티모달 시스템
```

## 🚀 실행 방법

### 1. 환경 설정

```bash
# 가상환경 생성 (권장)
python -m venv multimodal_env
source multimodal_env/bin/activate  # Linux/Mac
# 또는
multimodal_env\Scripts\activate     # Windows

# 패키지 설치
pip install -r requirements.txt
```

### 2. CLIP 실험 실행

```bash
python clip_experiment.py
```

**주요 기능:**
- Zero-shot 분류 성능 테스트
- 이미지-텍스트 검색 실험
- 프롬프트 엔지니어링 효과 검증

### 3. BLIP-2 데모 실행

```bash
python blip2_demo.py
```

**주요 기능:**
- 이미지 캡션 생성
- 질문-답변 시스템
- 상황 보고서 생성
- 교차 검증 테스트

### 4. 통합 멀티모달 시스템 실행

```bash
python autonomous_driving_multimodal.py
```

**주요 기능:**
- CLIP 기반 위험 요소 감지
- BLIP-2 상황 보고서 생성
- BEV + 언어 질의 시스템
- 안전 조치 트리거

## 🔬 실험 내용

### 1. CLIP 실험 (clip_experiment.py)

#### Zero-shot 분류
- **데이터셋**: 자율주행 관련 이미지 50장
- **클래스**: traffic light, pedestrian, car, traffic sign, construction
- **프롬프트 템플릿**:
  - `"a photo of a {class}."`
  - `"traffic scene with a {class}."`
  - 한국어 프롬프트

#### 이미지-텍스트 검색
- **쿼리**: "red traffic light", "pedestrian crossing", "construction cone"
- **평가 메트릭**: Recall@1, Recall@5, 평균 랭크

### 2. BLIP-2 실험 (blip2_demo.py)

#### 이미지 캡션 생성
- 자율주행 상황에 대한 자연어 설명 생성
- 교통 상황 분석 및 요약

#### 질문-답변 시스템
- 23개 자율주행 관련 질문에 대한 답변 생성
- 상황별 일관성 검증

### 3. 통합 시스템 (autonomous_driving_multimodal.py)

#### 위험 요소 감지
- **카테고리**: construction, pedestrian, emergency, accident, weather
- **임계값 기반** 위험도 평가 (LOW/MEDIUM/HIGH)

#### 상황 보고서 생성
- 자연어 기반 상황 요약
- 로그 시스템을 통한 기록 관리

#### BEV + 언어 질의
- 공간적 정보와 자연어 질의 결합
- "왼차선 막힘?", "보행자 위치?" 등 질의 처리

## 📊 예상 결과

### CLIP 실험 결과
| 프롬프트 유형 | 정확도 | 주요 특징 |
|---------------|--------|-----------|
| 기본 영어 | 72% | 일반적인 객체 인식 우수 |
| 맥락 포함 | 78% | 작은 객체(표지판) 인식률 향상 |
| 한국어 | 65% | 언어 간 성능 차이 존재 |

### 검색 성능
| 쿼리 | Recall@1 | Recall@5 | 평균 랭크 |
|------|----------|----------|-----------|
| red traffic light | 85% | 92% | 1.8 |
| pedestrian crossing | 72% | 88% | 2.3 |
| construction cone | 68% | 85% | 2.7 |

## 🚗 자율주행 적용 아이디어

### 1. CLIP 기반 위험구역 트리거
```python
# 위험 요소 프롬프트 예시
danger_prompts = [
    "construction cone on road",
    "pedestrian crossing street",
    "emergency vehicle with lights",
    "traffic accident scene"
]

# 임계값 초과 시 감속 명령
if max(clip_scores) > threshold:
    trigger_slowdown()
```

### 2. BLIP-2 상황 보고서
```
입력 이미지 → "좌측 차로에 정지차량, 보행자가 접근 중"
→ 의사결정 로그에 기록
→ 사고 분석 시 참고 자료
```

### 3. BEV + 언어 질의
```python
# 질의 예시
queries = [
    "Is the left lane blocked?",
    "Where are the pedestrians?",
    "Is there space to merge?"
]

# BEV + 언어 모델로 응답 생성
response = bev_language_model(bev_representation, query)
```

## ⚠️ 주의사항

### 하드웨어 요구사항
- **GPU**: CUDA 지원 GPU 권장 (최소 4GB VRAM)
- **RAM**: 최소 8GB (16GB 권장)
- **저장공간**: 최소 5GB (모델 다운로드 포함)

### 제약사항
- **실험용 코드**: 실제 자율주행에는 추가 검증 필요
- **더미 데이터**: 일부 기능은 시뮬레이션 데이터 사용
- **성능**: 실제 환경에서는 최적화 필요

## 🔍 문제 해결

### 일반적인 오류

1. **CUDA 메모리 부족**
   ```bash
   # CPU 모드로 실행
   export CUDA_VISIBLE_DEVICES=""
   ```

2. **모델 다운로드 실패**
   ```bash
   # 수동으로 모델 다운로드
   python -c "import clip; clip.load('ViT-B/32')"
   ```

3. **패키지 설치 오류**
   ```bash
   # pip 업그레이드
   pip install --upgrade pip
   pip install -r requirements.txt --force-reinstall
   ```


## 📖 참고 자료

### 논문
- [CLIP: Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- [BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://arxiv.org/abs/2301.12597)
- [BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers](https://arxiv.org/abs/2203.17270)

### 데이터셋
- [nuScenes](https://www.nuscenes.org/)
- [Waymo Open Dataset](https://waymo.com/open/)

### 도구
- [OpenAI CLIP](https://github.com/openai/CLIP)
- [LAVIS (BLIP-2)](https://github.com/salesforce/LAVIS)
- [BEVFormer](https://github.com/fundamentalvision/BEVFormer)

