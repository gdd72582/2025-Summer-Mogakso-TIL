#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MultimodalBasicTheory:
    """멀티모달 AI 기본 이론 실습 클래스"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 멀티모달 AI 기본 이론 실습 시작 (Device: {self.device})")
    
    def demonstrate_early_fusion(self, image_features, text_features):
        """Early Fusion (조기 융합) 시연"""
        print("\n📊 Early Fusion (조기 융합) 시연")
        print("=" * 50)
        
        # 원시 특징을 결합
        combined_features = torch.cat([image_features, text_features], dim=-1)
        
        print(f"이미지 특징 차원: {image_features.shape}")
        print(f"텍스트 특징 차원: {text_features.shape}")
        print(f"결합된 특징 차원: {combined_features.shape}")
        
        # 결합된 특징을 처리하는 간단한 네트워크
        fusion_network = nn.Sequential(
            nn.Linear(combined_features.shape[-1], 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        ).to(self.device)
        
        fused_output = fusion_network(combined_features)
        print(f"융합 후 출력 차원: {fused_output.shape}")
        
        return fused_output
    
    def demonstrate_late_fusion(self, image_features, text_features):
        """Late Fusion (후기 융합) 시연"""
        print("\n📊 Late Fusion (후기 융합) 시연")
        print("=" * 50)
        
        # 각 모달리티를 독립적으로 처리
        image_network = nn.Sequential(
            nn.Linear(image_features.shape[-1], 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        ).to(self.device)
        
        text_network = nn.Sequential(
            nn.Linear(text_features.shape[-1], 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        ).to(self.device)
        
        processed_image = image_network(image_features)
        processed_text = text_network(text_features)
        
        print(f"처리된 이미지 특징: {processed_image.shape}")
        print(f"처리된 텍스트 특징: {processed_text.shape}")
        
        # 후기 융합 (가중 평균)
        weights = torch.tensor([0.6, 0.4]).to(self.device)  # 이미지에 더 높은 가중치
        fused_output = weights[0] * processed_image + weights[1] * processed_text
        
        print(f"융합 후 출력 차원: {fused_output.shape}")
        print(f"가중치: 이미지={weights[0]:.1f}, 텍스트={weights[1]:.1f}")
        
        return fused_output
    
    def demonstrate_contrastive_learning(self, image_features, text_features):
        """Contrastive Learning (대조학습) 시연"""
        print("\n📊 Contrastive Learning (대조학습) 시연")
        print("=" * 50)
        
        # 특징 정규화
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # 유사도 계산 (코사인 유사도)
        similarity_matrix = torch.mm(image_features, text_features.T)
        
        print(f"유사도 행렬 크기: {similarity_matrix.shape}")
        print(f"대각선 요소 (정답 쌍): {torch.diag(similarity_matrix)}")
        
        # InfoNCE 손실 계산
        temperature = 0.07
        logits = similarity_matrix / temperature
        
        # 정답 라벨 (대각선)
        labels = torch.arange(logits.size(0)).to(self.device)
        
        # 교차 엔트로피 손실
        loss = F.cross_entropy(logits, labels)
        
        print(f"InfoNCE 손실: {loss.item():.4f}")
        print(f"온도 파라미터: {temperature}")
        
        return similarity_matrix, loss

def main():
    """메인 실행 함수"""
    print("🎓 멀티모달 AI 기본 이론 실습")
    print("=" * 60)
    
    # 실습 객체 생성
    theory_practice = MultimodalBasicTheory()
    
    # 더미 데이터 생성
    batch_size = 4
    image_dim = 512
    text_dim = 256
    
    image_features = torch.randn(batch_size, image_dim).to(theory_practice.device)
    text_features = torch.randn(batch_size, text_dim).to(theory_practice.device)
    
    print(f"생성된 더미 데이터:")
    print(f"이미지 특징: {image_features.shape}")
    print(f"텍스트 특징: {text_features.shape}")
    
    # 1. Early Fusion 시연
    early_fused = theory_practice.demonstrate_early_fusion(image_features, text_features)
    
    # 2. Late Fusion 시연
    late_fused = theory_practice.demonstrate_late_fusion(image_features, text_features)
    
    # 3. Contrastive Learning 시연
    similarity_matrix, contrastive_loss = theory_practice.demonstrate_contrastive_learning(
        image_features, text_features
    )
    
    print("\n✅ 멀티모달 AI 기본 이론 실습 완료!")

if __name__ == "__main__":
    main() 