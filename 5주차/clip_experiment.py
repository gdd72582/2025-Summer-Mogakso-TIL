#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import clip
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

class CLIPExperiment:
    def __init__(self):
        """CLIP 모델 초기화"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        
        # 자율주행 관련 클래스 정의
        self.classes = ["traffic light", "pedestrian", "car", "traffic sign", "construction"]
        
        # 프롬프트 템플릿 정의
        self.prompt_templates = {
            "basic": "a photo of a {}.",
            "context": "traffic scene with a {}.",
            "korean": "교통신호등이 있는 사진"  # 간단한 한국어 예시
        }
    
    def create_text_features(self, prompt_type="basic"):
        """텍스트 특징 생성"""
        if prompt_type == "korean":
            # 한국어는 간단한 예시로 처리
            text_inputs = torch.cat([clip.tokenize(self.prompt_templates["basic"].format(c)) for c in self.classes]).to(self.device)
        else:
            text_inputs = torch.cat([clip.tokenize(self.prompt_templates[prompt_type].format(c)) for c in self.classes]).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.encode_text(text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        
        return text_features
    
    def predict_single_image(self, image_path, prompt_type="basic"):
        """단일 이미지 예측"""
        # 이미지 전처리
        image = Image.open(image_path).convert('RGB')
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # 텍스트 특징 생성
        text_features = self.create_text_features(prompt_type)
        
        # 이미지 특징 생성
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        
        # 유사도 계산
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        
        # 예측 결과
        predicted_class = self.classes[similarity.argmax().item()]
        confidence = similarity.max().item()
        
        return predicted_class, confidence, similarity.cpu().numpy()[0]
    
    def zero_shot_classification(self, image_paths, prompt_type="basic"):
        """Zero-shot 분류 실험"""
        results = []
        text_features = self.create_text_features(prompt_type)
        
        for image_path in image_paths:
            try:
                image = Image.open(image_path).convert('RGB')
                image_input = self.preprocess(image).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    image_features = self.model.encode_image(image_input)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                predicted_class = self.classes[similarity.argmax().item()]
                confidence = similarity.max().item()
                
                results.append({
                    'image_path': image_path,
                    'predicted': predicted_class,
                    'confidence': confidence,
                    'similarities': similarity.cpu().numpy()[0]
                })
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
        
        return results
    
    def image_text_search(self, image_paths, query_texts):
        """이미지-텍스트 검색 실험"""
        # 쿼리 텍스트 특징 생성
        query_inputs = torch.cat([clip.tokenize(query) for query in query_texts]).to(self.device)
        with torch.no_grad():
            query_features = self.model.encode_text(query_inputs)
            query_features /= query_features.norm(dim=-1, keepdim=True)
        
        # 이미지 특징 생성
        image_features_list = []
        valid_image_paths = []
        
        for image_path in image_paths:
            try:
                image = Image.open(image_path).convert('RGB')
                image_input = self.preprocess(image).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    image_features = self.model.encode_image(image_input)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                
                image_features_list.append(image_features)
                valid_image_paths.append(image_path)
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
        
        if not image_features_list:
            return []
        
        # 모든 이미지 특징을 하나로 결합
        all_image_features = torch.cat(image_features_list, dim=0)
        
        # 유사도 계산
        similarity_matrix = (100.0 * all_image_features @ query_features.T)
        
        # 각 쿼리에 대한 상위 이미지 찾기
        search_results = []
        for i, query in enumerate(query_texts):
            similarities = similarity_matrix[:, i]
            top_indices = similarities.argsort(descending=True)
            
            query_results = []
            for j, idx in enumerate(top_indices[:5]):  # Top-5
                query_results.append({
                    'rank': j + 1,
                    'image_path': valid_image_paths[idx.item()],
                    'similarity': similarities[idx].item()
                })
            
            search_results.append({
                'query': query,
                'results': query_results
            })
        
        return search_results
    
    def visualize_results(self, results, title="CLIP Zero-shot Classification Results"):
        """결과 시각화"""
        if not results:
            print("No results to visualize")
            return
        
        # 정확도 계산 (간단한 예시)
        total = len(results)
        print(f"\n{title}")
        print(f"Total images processed: {total}")
        
        # 클래스별 예측 분포
        class_counts = {}
        for result in results:
            pred = result['predicted']
            class_counts[pred] = class_counts.get(pred, 0) + 1
        
        print("\nPrediction distribution:")
        for class_name, count in class_counts.items():
            percentage = (count / total) * 100
            print(f"  {class_name}: {count} ({percentage:.1f}%)")
        
        # 평균 신뢰도
        avg_confidence = np.mean([r['confidence'] for r in results])
        print(f"\nAverage confidence: {avg_confidence:.3f}")

def main():
    """메인 실행 함수"""
    print("🚗 CLIP 자율주행 실험 시작")
    
    # CLIP 실험 객체 생성
    experiment = CLIPExperiment()
    
    # 예시 이미지 경로 (실제로는 자율주행 관련 이미지 사용)
    # 여기서는 샘플 이미지 경로를 가정
    sample_images = [
        "sample_traffic_light.jpg",
        "sample_pedestrian.jpg", 
        "sample_car.jpg",
        "sample_sign.jpg",
        "sample_construction.jpg"
    ]
    
    # 쿼리 텍스트 정의
    query_texts = [
        "red traffic light",
        "pedestrian crossing", 
        "construction cone"
    ]
    
    print("\n1. Zero-shot 분류 실험")
    print("=" * 50)
    
    # 기본 프롬프트로 분류
    basic_results = experiment.zero_shot_classification(sample_images, "basic")
    experiment.visualize_results(basic_results, "Basic Prompt Results")
    
    # 맥락 포함 프롬프트로 분류
    context_results = experiment.zero_shot_classification(sample_images, "context")
    experiment.visualize_results(context_results, "Context Prompt Results")
    
    print("\n2. 이미지-텍스트 검색 실험")
    print("=" * 50)
    
    # 검색 실험
    search_results = experiment.image_text_search(sample_images, query_texts)
    
    for query_result in search_results:
        print(f"\nQuery: '{query_result['query']}'")
        print("Top results:")
        for result in query_result['results']:
            print(f"  Rank {result['rank']}: {result['image_path']} (similarity: {result['similarity']:.3f})")
    
    print("\n✅ 실험 완료!")

if __name__ == "__main__":
    main() 