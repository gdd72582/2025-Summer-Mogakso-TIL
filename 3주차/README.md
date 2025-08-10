# 3주차 - 자율주행 경로 계획 및 제어 시스템

## 개요
이 주차에서는 자율주행 차량의 경로 계획(Path Planning)과 제어(Control) 시스템을 구현합니다. Pure Pursuit 알고리즘을 기반으로 한 경로 추종과 Lattice Planner를 이용한 장애물 회피 시스템을 다룹니다.

## 파일 구조

### 1. `global_path_pub.py`
- **기능**: 전역 경로(Global Path) 발행 노드
- **주요 내용**:
  - `kcity.txt` 파일에서 경로 데이터를 읽어옴
  - `/global_path` 토픽으로 전역 경로를 발행
  - 20Hz 주기로 경로 데이터를 지속적으로 발행

### 2. `gpsimu_parser.py`
- **기능**: GPS/IMU 센서 데이터 파싱 및 변환
- **주요 내용**:
  - GPS 위도/경도 데이터를 UTM 좌표계로 변환
  - IMU 센서의 방향 데이터 처리
  - `/odom` 토픽으로 변환된 위치 정보 발행
  - UTM Zone 52 기준으로 좌표 변환

### 3. `local_path_pub.py`
- **기능**: 지역 경로(Local Path) 생성 및 발행
- **주요 내용**:
  - 현재 차량 위치에서 가장 가까운 전역 경로 웨이포인트 찾기
  - 현재 위치 기준으로 50개 웨이포인트로 구성된 지역 경로 생성
  - `/local_path` 토픽으로 지역 경로 발행

### 4. `lattice_planner.py`
- **기능**: 격자 기반 경로 계획 및 장애물 회피
- **주요 내용**:
  - 장애물 감지 및 충돌 검사
  - 다중 경로 생성 (Lattice Path)
  - 비용 기반 최적 경로 선택
  - 장애물이 있을 때 회피 경로 생성, 없을 때 원래 경로 사용

### 5. `pure_pursuit_pid_velocity_planning.py`
- **기능**: Pure Pursuit 알고리즘 기반 차량 제어
- **주요 내용**:
  - Look Ahead Distance 기반 조향각 계산
  - PID 제어를 통한 속도 제어
  - 곡률 기반 속도 계획
  - 차량의 조향 및 가속/제동 명령 생성

## 핵심 알고리즘

### Pure Pursuit 알고리즘
- 차량의 현재 위치에서 일정 거리(Look Ahead Distance) 앞의 경로점을 찾아 조향각 계산
- 속도에 비례하여 Look Ahead Distance 조정
- PID 제어를 통한 정확한 속도 추종

### Lattice Planner
- 장애물 감지 시 다중 경로 생성
- 각 경로의 충돌 위험도 계산
- 최소 비용 경로 선택으로 안전한 회피 수행

## 시스템 구조
```
GPS/IMU → gpsimu_parser → odom
    ↓
global_path_pub → global_path
    ↓
local_path_pub → local_path
    ↓
lattice_planner → lattice_path
    ↓
pure_pursuit_pid_velocity_planning → ctrl_cmd
```

## 실행 방법
1. `global_path_pub.py` 실행하여 전역 경로 발행
2. `gpsimu_parser.py` 실행하여 센서 데이터 처리
3. `local_path_pub.py` 실행하여 지역 경로 생성
4. `lattice_planner.py` 실행하여 장애물 회피 경로 계획
5. `pure_pursuit_pid_velocity_planning.py` 실행하여 차량 제어

## 주요 토픽
- `/global_path`: 전역 경로 데이터
- `/odom`: 차량 위치 및 방향 정보
- `/local_path`: 지역 경로 데이터
- `/lattice_path`: 장애물 회피 경로
- `/ctrl_cmd`: 차량 제어 명령 (조향, 가속, 제동) 