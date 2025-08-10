# 1주차: ROS 기본 개념과 자율주행 시스템 기초 개념

## ROS (Robot Operating System) 기본 개념

### 1. ROS란?
ROS: 로봇 소프트웨어를 개발하기 위한 소프트웨어 프레임워크
* 노드간 통신을 바탕으로 전체 시스템 구동
* 메시지 기록 재생 기능으로 반복적인 실험 가능, 알고리즘 개발 용이

---

### 2. ROS의 주요 구성 요소

| 구성 요소 | 설명 |
|-----------|------|
| **ROS Master**  | 노드와 노드 사이의 연결과 통신을 위한 서버, `roscore`|
| **Node**  | ROS에서 실행되는 최소 단위 프로세스, 하나의 목적에 하나의 노드를 개발하는 것을 추천 |
| **Topic** | 노드 간의 비동기 메시지 통신 채널, 메시지를 송신하기 위해 토픽으로 마스터에 등록하여 메시지를 보냄 |
| **Message** | Topic으로 송수신되는 데이터 구조 |
| **Publisher / Subscriber** | 메시지를 발행(Pub) / 구독(Sub)하는 통신 구조 |
| **Service** | 요청-응답(동기) 방식의 노드 간 통신 -> 양방향 |
| **Parameter Server** | ROS 전체에서 공유 가능한 전역 파라미터 저장소 |
| **Launch File** | 여러 노드를 한 번에 실행시키는 XML 형식 설정 파일 |

---

### 3. ROS 통신 방식

**Publisher/Subscriber 패턴:**
- Publisher는 메시지를 송신하고, Subscriber는 메시지를 수신하는 단방향, 비동기 통신
- 일대다, 일대일, 다대다 통신이 모두 가능
- Topic을 통해 통신

**Request/Response 패턴 (Service):**
- 서버와 클라이언트 간의 양방향, 동기통신
- 요청에 대한 즉시 응답이 필요한 경우 사용

---

### 4. ROS 파일 시스템

- **Package (패키지):** 최소 빌드 단위, 배포 단위
- **Workspace (워크스페이스):** 전체를 포함하는 작업 공간
- **Launch file (런치 파일):** 여러 파일을 동시에 실행하는 설정 파일

---

### 5. ROS 명령어 정리


#### 5.1. ROS 셸 명령어
| 명령어 | 기능 |
|--------|------|
| **roscd** | 지정한 ROS 패키지의 디렉토리 위치로 이동 |
| **rosls** | ROS 패키지의 파일 목록 확인 |

#### 5.2. ROS 실행 명령어
| 명령어 | 기능 |
|--------|------|
| **roscore** | 마스터 노드 실행 |
| **rosrun** | 노드 실행 |
| **roslaunch** | 여러 노드 실행 및 실행 옵션 설정 |
| **rosclean** | ROS log 파일 검사 및 삭제 |

#### 5.3. ROS 정보 명령어
| 명령어 | 기능 |
|--------|------|
| **rostopic** | ROS 토픽 정보 확인 |
| **rosservice** | ROS 서비스 정보 확인 |
| **rosnode** | ROS 노드 정보 확인 |
| **rosbag** | ROS 메시지 기록, 재생 |
| **rosmsg** | ROS 메시지 파일 정보 확인 |
| **rossrv** | ROS 서비스 파일 정보 확인 |
| **rosversion** | ROS 패키지 및 릴리즈 버전 정보 확인 |

#### 5.4. ROS Catkin 명령어
| 명령어 | 기능 |
|--------|------|
| **catkin_create_pkg** | Catkin 빌드 시스템으로 패키지 자동 생성 |
| **catkin_make** | Catkin 빌드 시스템에 기반을 둔 빌드 |
| **catkin_init_workspace** | Catkin 빌드 시스템 작업 폴더 초기화 |

---

## 자율주행 시스템 기초 개념

### 6. 자율주행의 정의

자율주행이란 운전자의 개입 없이 차량이 스스로 주행하는 기술을 의미합니다.

**SAE 자율주행 레벨:**
- **Level 0:** 완전 수동 주행
- **Level 1:** 운전자 지원 (조향 또는 가속/제동 중 하나만 자동화)
- **Level 2:** 부분 자동화 (조향과 가속/제동 동시 자동화)
- **Level 3:** 조건부 자동화 (특정 조건에서 완전 자동화, 비상시 운전자 개입)
- **Level 4:** 고도 자동화 (특정 환경에서 완전 자동화)
- **Level 5:** 완전 자동화 (모든 환경에서 완전 자동화)

---

### 7. 자율주행 핵심 기술 단계

자율주행은 크게 **인지 → 판단 → 제어**의 3단계 프로세스로 구성됩니다.

#### 7.1 인지 (Perception)
**목표:** 차량 주변의 환경을 인식하고 필요한 정보를 추출

**기술 요소:**
- **정적 객체 인식:** 차로, 차선, 횡단보도, 터널, 고가도로 등
- **동적 객체 인식:** 차량, 보행자, 신호등, 이동 장애물 등
- **객체 탐지(Object Detection):**
  - **Classification:** 객체 종류 분류
  - **Localization:** 위치(Bounding Box) 추정
  - **Object Detection:** 다수 객체에 대해 분류 + 위치 동시 추정
- **딥러닝 기반 알고리즘 예시:**
  - **YOLO:** 실시간 객체 탐지
  - **Faster R-CNN:** 정확도 중심
  - **DETR:** Transformer 기반 최신 탐지
- **정밀도로지도(HD Map) 활용:**
  - 포인트클라우드맵(PointCloud Map) → 3D 점군 데이터
  - 벡터맵(Vector Map) → 차선·신호등 등 의미 정보
  - 격자맵(Grid Map) → 경로 계획에 활용
- **주요 센서:**
  - 카메라: 고해상도 인식, 표지판/문자 인식에 강점
  - LiDAR: 3D 거리·형태 인식, 정밀도 높음
  - RADAR: 거리·속도 인식, 날씨 영향 적음

#### 7.2 판단 (Decision Making)
**목표:** 인지 정보를 기반으로 주행 전략과 경로를 계획

**기술 요소:**
- **경로 계획(Path Planning):**
  - **전역 경로 계획(Global Path Planning):** 전체 지도 기반 최적 경로 생성
  - **지역 경로 계획(Local Path Planning):** 실시간 장애물 회피·대응
  - **궤적 생성(Trajectory Generation):**
    - 단순 1차 함수 → 급격한 조향 변화
    - 3차 함수 및 스플라인(Spline) → 부드러운 주행
- **경로 추종(Path Tracking):**
  - **Pure Pursuit 알고리즘:**
    - 전방주시거리(Look-ahead distance) 기반 조향각 계산
    - 자전거 모델(Bicycle Model) 활용
- **속도 계획:**
  - Adaptive Cruise Control(ACC)
  - Autonomous Emergency Braking(AEB)

---

### 7.3 경로 계획 (Path Planning)

#### 7.3.1 전역 경로 계획 (Global Path Planning)
- 차량을 직접 주행시키며 위치 좌표(X, Y)를 기록 → 웨이포인트 기반 경로 생성
- 기록된 경로를 모의주행 또는 자동 경로 생성 알고리즘에 활용 가능
- **웨이포인트 형식:**
```
X         Y
144.4321  2312.2344
147.2140  2314.6561
150.6157  2317.3059
```

#### 7.3.2 지역 경로 계획 (Local Path Planning)
- 실시간 센서 데이터를 활용하여 예상치 못한 장애물 회피 경로 생성
- 전역 경로 기반이지만, 주행 중 동적으로 수정

#### 7.3.3 궤적 생성 (Trajectory Generation)
- **1차 함수 궤적:**
  - 시작점과 끝점을 직선으로 연결
  - 급격한 조향각 변화 → 차량 거동에 부정적 영향
- **3차 함수 + Spline 궤적:**
  - 시작점과 끝점을 매끄럽게 연결
  - 조향 입력 변화가 부드러워짐 → 안정성 향상
  - 현재 자율주행 차량에서 가장 널리 사용

---

### 7.4 경로 추종 (Path Tracking)

#### 7.4.1 Pure Pursuit 알고리즘
- 경로 위의 한 점을 원호를 그리며 추종하는 방식
- **전방주시거리(Look-ahead distance)**를 이용해 조향각 계산
- Ackermann 조향 기하구조를 단순화한 **Bicycle 모델** 사용
- **계산식:**

$$
\sin(\alpha) = \frac{d_{la} / 2}{R}
$$

$$
R = \frac{d_{la}}{2\sin(\alpha)}
$$

$$
\delta = \tan^{-1} \left( \frac{2L \sin(\alpha)}{k v_x} \right)
$$

- \( L \): Wheel base  
- \( \delta \): Steering angle  
- \( R \): 곡률 반경  
- \( d_{la} \): 전방주시거리  
- \( v_x \): 종방향 속도  
- \( k \): Look-ahead 비율 계수

---

### 7.5 속도 제어 (Speed Control)

#### 7.5.1 Adaptive Cruise Control (ACC)
- 전방 차량과의 거리 및 상대 속도를 측정하여 가·감속 제어
- 안전 거리 확보를 위해 속도를 자동 조정

#### 7.5.2 Autonomous Emergency Braking (AEB)
- 돌발 장애물 발생 시 긴급 제동 수행
- 센서(카메라, RADAR) 기반 충돌 위험 예측 후 즉각 제동

#### 7.5.3 PID 제어 (Proportional-Integral-Derivative)
- **P (비례)**: 현재 오차 크기에 비례한 제어량
- **I (적분)**: 누적 오차 제거
- **D (미분)**: 급격한 변화 억제 → 안정성 향상
- **제어식:**

$$
MV(t) = K_P e(t) + K_I \int_{0}^{t} e(\tau) \, d\tau + K_D \frac{de(t)}{dt}
$$

- \( K_P \): 비례 게인 (Proportional gain)  
- \( K_I \): 적분 게인 (Integral gain)  
- \( K_D \): 미분 게인 (Derivative gain)  
- \( e(t) \): 시간 \( t \)에서의 오차
- **주행에서의 활용:**
  - ACC의 속도 제어
  - 곡선 주행 시 안정적인 조향



---
*이 문서는 1주차 ROS 기본 개념과 자율주행 시스템 기초 개념 학습 내용을 정리한 문서입니다.* 