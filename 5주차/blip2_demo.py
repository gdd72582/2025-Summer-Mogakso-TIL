#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration

class BLIP2Demo:
    def __init__(self):
        """BLIP-2 모델 초기화"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # BLIP-2 모델 로드 (가벼운 버전 사용)
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b", 
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        
        # 자율주행 관련 질문 템플릿
        self.autonomous_driving_questions = [
            "What traffic signs are visible in this image?",
            "Are there any pedestrians in the scene?",
            "What is the traffic light status?",
            "Are there any obstacles on the road?",
            "What is the weather condition?",
            "Is this an intersection or a straight road?",
            "Are there any emergency vehicles?",
            "What is the road condition?",
            "Are there any construction zones?",
            "What is the speed limit indicated?",
            "Are there any lane markings visible?",
            "What type of vehicles are present?",
            "Is this a residential or highway area?",
            "Are there any crosswalks?",
            "What is the time of day?",
            "Are there any traffic cones or barriers?",
            "What is the road surface condition?",
            "Are there any parking signs?",
            "What is the traffic density?",
            "Are there any school zones?",
            "What is the road curvature?",
            "Are there any bus stops?",
            "What is the visibility condition?"
        ]
    
    def generate_caption(self, image_path):
        """이미지 캡션 생성"""
        try:
            # 이미지 로드
            if image_path.startswith('http'):
                image = Image.open(requests.get(image_path, stream=True).raw).convert('RGB')
            else:
                image = Image.open(image_path).convert('RGB')
            
            # 입력 처리
            inputs = self.processor(image, return_tensors="pt").to(self.device, torch.float16)
            
            # 캡션 생성
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_length=50,
                    num_beams=5,
                    early_stopping=True
                )
            
            # 결과 디코딩
            caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            
            return caption
            
        except Exception as e:
            return f"Error generating caption: {e}"
    
    def answer_question(self, image_path, question):
        """질문에 대한 답변 생성"""
        try:
            # 이미지 로드
            if image_path.startswith('http'):
                image = Image.open(requests.get(image_path, stream=True).raw).convert('RGB')
            else:
                image = Image.open(image_path).convert('RGB')
            
            # 입력 처리
            inputs = self.processor(image, question, return_tensors="pt").to(self.device, torch.float16)
            
            # 답변 생성
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_length=30,
                    num_beams=5,
                    early_stopping=True
                )
            
            # 결과 디코딩
            answer = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            
            return answer
            
        except Exception as e:
            return f"Error generating answer: {e}"
    
    def generate_situation_report(self, image_path):
        """자율주행 상황 보고서 생성"""
        try:
            # 기본 캡션 생성
            basic_caption = self.generate_caption(image_path)
            
            # 주요 질문들에 대한 답변 생성
            key_questions = [
                "What traffic signs are visible in this image?",
                "Are there any pedestrians in the scene?",
                "What is the traffic light status?",
                "Are there any obstacles on the road?"
            ]
            
            answers = {}
            for question in key_questions:
                answer = self.answer_question(image_path, question)
                answers[question] = answer
            
            # 상황 보고서 구성
            report = {
                "basic_caption": basic_caption,
                "traffic_signs": answers.get("What traffic signs are visible in this image?", "None detected"),
                "pedestrians": answers.get("Are there any pedestrians in the scene?", "None detected"),
                "traffic_light": answers.get("What is the traffic light status?", "Not visible"),
                "obstacles": answers.get("Are there any obstacles on the road?", "None detected")
            }
            
            return report
            
        except Exception as e:
            return {"error": f"Error generating situation report: {e}"}
    
    def cross_validation_test(self, image_path):
        """교차 검증 테스트 - 일관성 점검"""
        print(f"🔍 교차 검증 테스트: {image_path}")
        print("=" * 60)
        
        # 기본 캡션
        caption = self.generate_caption(image_path)
        print(f"📝 기본 캡션: {caption}")
        print()
        
        # 관련 질문들로 일관성 테스트
        related_questions = [
            "What traffic signs are visible in this image?",
            "Are there any pedestrians in the scene?",
            "What is the traffic light status?",
            "Are there any obstacles on the road?",
            "What is the weather condition?",
            "Is this an intersection or a straight road?"
        ]
        
        print("❓ 질문-답변 테스트:")
        for i, question in enumerate(related_questions, 1):
            answer = self.answer_question(image_path, question)
            print(f"  {i}. Q: {question}")
            print(f"     A: {answer}")
            print()
        
        # 상황 보고서
        report = self.generate_situation_report(image_path)
        print("📊 상황 보고서:")
        for key, value in report.items():
            if key != "basic_caption":
                print(f"  • {key}: {value}")
        
        print("\n" + "=" * 60)
        return report

def main():
    """메인 실행 함수"""
    print("🚗 BLIP-2 자율주행 상황 분석 시작")
    
    # BLIP-2 데모 객체 생성
    blip2_demo = BLIP2Demo()
    
    # 예시 이미지 경로들 (실제로는 자율주행 관련 이미지 사용)
    sample_images = [
        "sample_intersection.jpg",
        "sample_pedestrian_crossing.jpg",
        "sample_traffic_light.jpg",
        "sample_construction_zone.jpg",
        "sample_highway.jpg"
    ]
    
    print("\n1. 이미지 캡션 생성 테스트")
    print("=" * 50)
    
    for i, image_path in enumerate(sample_images[:2], 1):  # 처음 2개만 테스트
        print(f"\n이미지 {i}: {image_path}")
        caption = blip2_demo.generate_caption(image_path)
        print(f"캡션: {caption}")
    
    print("\n2. 질문-답변 테스트")
    print("=" * 50)
    
    # 특정 이미지에 대한 질문-답변 테스트
    test_image = sample_images[0]
    test_questions = [
        "What traffic signs are visible in this image?",
        "Are there any pedestrians in the scene?",
        "What is the traffic light status?"
    ]
    
    for question in test_questions:
        answer = blip2_demo.answer_question(test_image, question)
        print(f"Q: {question}")
        print(f"A: {answer}")
        print()
    
    print("\n3. 상황 보고서 생성 테스트")
    print("=" * 50)
    
    # 상황 보고서 생성
    report = blip2_demo.generate_situation_report(test_image)
    print("상황 보고서:")
    for key, value in report.items():
        print(f"  {key}: {value}")
    
    print("\n4. 교차 검증 테스트")
    print("=" * 50)
    
    # 교차 검증 테스트 (첫 번째 이미지)
    blip2_demo.cross_validation_test(sample_images[0])
    
    print("\n✅ BLIP-2 실험 완료!")

if __name__ == "__main__":
    main() 