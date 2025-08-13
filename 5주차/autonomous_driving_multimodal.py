#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
from PIL import Image
import json
from datetime import datetime
from typing import Dict, List, Tuple

class AutonomousDrivingMultimodal:
    def __init__(self):
        """자율주행 멀티모달 시스템 초기화"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 위험 요소 프롬프트 정의
        self.danger_prompts = {
            "construction": [
                "construction cone on road",
                "construction zone ahead",
                "road work in progress",
                "traffic cone blocking lane"
            ],
            "pedestrian": [
                "pedestrian crossing street",
                "person walking on road",
                "pedestrian near intersection",
                "people crossing"
            ],
            "emergency": [
                "emergency vehicle with lights",
                "police car with flashing lights",
                "ambulance on road",
                "fire truck emergency"
            ],
            "accident": [
                "traffic accident scene",
                "car crash on road",
                "vehicle collision",
                "road accident"
            ],
            "weather": [
                "heavy rain on road",
                "snow covered road",
                "foggy road conditions",
                "icy road surface"
            ]
        }
        
        # 위험도 임계값 설정
        self.danger_thresholds = {
            "construction": 0.7,
            "pedestrian": 0.8,
            "emergency": 0.9,
            "accident": 0.85,
            "weather": 0.75
        }
        
        # 시스템 상태
        self.current_speed = 0.0
        self.safety_mode = False
        self.situation_log = []
    
    def clip_danger_detection(self, image_path: str, clip_model) -> Dict:
        """CLIP 기반 위험 요소 감지"""
        try:
            # 이미지 전처리
            image = Image.open(image_path).convert('RGB')
            
            # 각 위험 카테고리별 스코어 계산
            danger_scores = {}
            max_scores = {}
            
            for category, prompts in self.danger_prompts.items():
                scores = []
                for prompt in prompts:
                    # CLIP을 사용한 유사도 계산 (간단한 구현)
                    score = self._calculate_clip_similarity(clip_model, image, prompt)
                    scores.append(score)
                
                danger_scores[category] = scores
                max_scores[category] = max(scores)
            
            # 위험도 평가
            danger_level = self._evaluate_danger_level(max_scores)
            
            return {
                "danger_scores": danger_scores,
                "max_scores": max_scores,
                "danger_level": danger_level,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Error in danger detection: {e}"}
    
    def _calculate_clip_similarity(self, clip_model, image, prompt):
        """CLIP 유사도 계산 (간단한 구현)"""
        # 실제로는 CLIP 모델을 사용하여 계산
        # 여기서는 더미 값 반환
        return np.random.uniform(0.1, 0.9)
    
    def _evaluate_danger_level(self, max_scores: Dict) -> str:
        """위험도 수준 평가"""
        high_danger_count = 0
        
        for category, score in max_scores.items():
            threshold = self.danger_thresholds.get(category, 0.7)
            if score > threshold:
                high_danger_count += 1
        
        if high_danger_count >= 3:
            return "HIGH"
        elif high_danger_count >= 1:
            return "MEDIUM"
        else:
            return "LOW"
    
    def blip2_situation_report(self, image_path: str, blip2_model) -> Dict:
        """BLIP-2 기반 상황 보고서 생성"""
        try:
            # BLIP-2를 사용한 상황 분석 (간단한 구현)
            situation_analysis = {
                "traffic_signs": "stop sign visible",
                "pedestrians": "no pedestrians detected",
                "traffic_light": "green light",
                "obstacles": "clear road ahead",
                "weather": "clear weather",
                "road_condition": "good road surface"
            }
            
            # 자연어 요약 생성
            summary = self._generate_situation_summary(situation_analysis)
            
            return {
                "situation_analysis": situation_analysis,
                "summary": summary,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Error in situation report: {e}"}
    
    def _generate_situation_summary(self, analysis: Dict) -> str:
        """상황 요약 생성"""
        summary_parts = []
        
        if analysis.get("traffic_signs"):
            summary_parts.append(f"Traffic signs: {analysis['traffic_signs']}")
        
        if analysis.get("pedestrians") != "no pedestrians detected":
            summary_parts.append(f"Pedestrians: {analysis['pedestrians']}")
        
        if analysis.get("traffic_light"):
            summary_parts.append(f"Traffic light: {analysis['traffic_light']}")
        
        if analysis.get("obstacles") != "clear road ahead":
            summary_parts.append(f"Obstacles: {analysis['obstacles']}")
        
        summary_parts.append(f"Weather: {analysis.get('weather', 'unknown')}")
        summary_parts.append(f"Road condition: {analysis.get('road_condition', 'unknown')}")
        
        return ". ".join(summary_parts) + "."
    
    def bev_language_query(self, bev_representation: np.ndarray, query: str) -> Dict:
        """BEV + 언어 질의 시스템"""
        try:
            # BEV 표현에서 공간적 정보 추출 (간단한 구현)
            spatial_info = self._extract_spatial_info(bev_representation)
            
            # 질의에 따른 응답 생성
            response = self._process_spatial_query(query, spatial_info)
            
            return {
                "query": query,
                "response": response,
                "spatial_info": spatial_info,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Error in BEV language query: {e}"}
    
    def _extract_spatial_info(self, bev_representation: np.ndarray) -> Dict:
        """BEV 표현에서 공간적 정보 추출"""
        # 간단한 공간적 정보 추출 (실제로는 더 복잡한 처리 필요)
        return {
            "left_lane_blocked": np.random.choice([True, False], p=[0.3, 0.7]),
            "right_lane_blocked": np.random.choice([True, False], p=[0.2, 0.8]),
            "pedestrians_present": np.random.choice([True, False], p=[0.4, 0.6]),
            "traffic_density": np.random.uniform(0.1, 0.9),
            "merge_space_available": np.random.choice([True, False], p=[0.6, 0.4])
        }
    
    def _process_spatial_query(self, query: str, spatial_info: Dict) -> str:
        """공간적 질의 처리"""
        query_lower = query.lower()
        
        if "left lane" in query_lower and "blocked" in query_lower:
            return "Yes" if spatial_info["left_lane_blocked"] else "No"
        
        elif "right lane" in query_lower and "blocked" in query_lower:
            return "Yes" if spatial_info["right_lane_blocked"] else "No"
        
        elif "pedestrian" in query_lower:
            return "Yes" if spatial_info["pedestrians_present"] else "No"
        
        elif "merge" in query_lower and "space" in query_lower:
            return "Yes" if spatial_info["merge_space_available"] else "No"
        
        elif "traffic" in query_lower and "density" in query_lower:
            density = spatial_info["traffic_density"]
            if density < 0.3:
                return "Low traffic density"
            elif density < 0.7:
                return "Medium traffic density"
            else:
                return "High traffic density"
        
        else:
            return "Query not understood"
    
    def trigger_safety_action(self, danger_level: str, situation_report: Dict) -> Dict:
        """안전 조치 트리거"""
        actions = []
        
        if danger_level == "HIGH":
            actions.extend([
                "Immediate speed reduction",
                "Emergency braking preparation",
                "Hazard warning activation",
                "Driver alert system activation"
            ])
            self.safety_mode = True
            
        elif danger_level == "MEDIUM":
            actions.extend([
                "Moderate speed reduction",
                "Increased vigilance mode",
                "Lane change restriction"
            ])
            
        else:  # LOW
            actions.extend([
                "Normal driving mode",
                "Regular monitoring"
            ])
            self.safety_mode = False
        
        # 상황 로그에 기록
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "danger_level": danger_level,
            "situation_summary": situation_report.get("summary", ""),
            "actions_taken": actions,
            "safety_mode": self.safety_mode
        }
        self.situation_log.append(log_entry)
        
        return {
            "actions": actions,
            "safety_mode": self.safety_mode,
            "log_entry": log_entry
        }
    
    def get_system_status(self) -> Dict:
        """시스템 상태 반환"""
        return {
            "current_speed": self.current_speed,
            "safety_mode": self.safety_mode,
            "log_entries_count": len(self.situation_log),
            "last_update": datetime.now().isoformat()
        }
    
    def save_situation_log(self, filename: str = "autonomous_driving_log.json"):
        """상황 로그 저장"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.situation_log, f, indent=2, ensure_ascii=False)
            print(f"✅ 상황 로그가 {filename}에 저장되었습니다.")
        except Exception as e:
            print(f"❌ 로그 저장 실패: {e}")

def main():
    """메인 실행 함수"""
    print("🚗 자율주행 멀티모달 시스템 시작")
    
    # 시스템 초기화
    system = AutonomousDrivingMultimodal()
    
    # 예시 이미지 경로
    sample_image = "sample_traffic_scene.jpg"
    
    print("\n1. CLIP 기반 위험 요소 감지")
    print("=" * 50)
    
    # CLIP 모델 시뮬레이션 (실제로는 CLIP 모델 로드)
    clip_model = None  # 실제로는 CLIP 모델 객체
    
    danger_result = system.clip_danger_detection(sample_image, clip_model)
    print(f"위험도 수준: {danger_result.get('danger_level', 'UNKNOWN')}")
    print(f"최대 스코어: {danger_result.get('max_scores', {})}")
    
    print("\n2. BLIP-2 상황 보고서 생성")
    print("=" * 50)
    
    # BLIP-2 모델 시뮬레이션
    blip2_model = None  # 실제로는 BLIP-2 모델 객체
    
    situation_report = system.blip2_situation_report(sample_image, blip2_model)
    print(f"상황 요약: {situation_report.get('summary', '')}")
    
    print("\n3. BEV + 언어 질의 테스트")
    print("=" * 50)
    
    # BEV 표현 시뮬레이션
    bev_representation = np.random.rand(100, 100, 3)  # 더미 BEV 데이터
    
    queries = [
        "Is the left lane blocked?",
        "Where are the pedestrians?",
        "Is there space to merge?",
        "What is the traffic density?"
    ]
    
    for query in queries:
        response = system.bev_language_query(bev_representation, query)
        print(f"Q: {query}")
        print(f"A: {response.get('response', '')}")
        print()
    
    print("\n4. 안전 조치 트리거")
    print("=" * 50)
    
    danger_level = danger_result.get('danger_level', 'LOW')
    safety_actions = system.trigger_safety_action(danger_level, situation_report)
    
    print(f"위험도: {danger_level}")
    print(f"안전 모드: {safety_actions['safety_mode']}")
    print("조치 사항:")
    for action in safety_actions['actions']:
        print(f"  • {action}")
    
    print("\n5. 시스템 상태 확인")
    print("=" * 50)
    
    status = system.get_system_status()
    print(f"현재 속도: {status['current_speed']} km/h")
    print(f"안전 모드: {status['safety_mode']}")
    print(f"로그 항목 수: {status['log_entries_count']}")
    
    # 로그 저장
    system.save_situation_log()
    
    print("\n✅ 자율주행 멀티모달 시스템 테스트 완료!")

if __name__ == "__main__":
    main() 