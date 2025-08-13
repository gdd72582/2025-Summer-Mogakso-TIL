#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class MultimodalVisualization:
    """멀티모달 AI 시각화 및 분석 클래스"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"📊 멀티모달 AI 시각화 시작 (Device: {self.device})")
    
    def visualize_fusion_comparison(self, early_fused, late_fused):
        """융합 방법 비교 시각화"""
        print("\n📊 융합 방법 비교 시각화")
        print("=" * 50)
        
        # 특징 분포 비교
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Early Fusion 분포
        early_data = early_fused.detach().cpu().numpy().flatten()
        axes[0].hist(early_data, bins=30, alpha=0.7, color='blue')
        axes[0].set_title('Early Fusion 특징 분포')
        axes[0].set_xlabel('특징 값')
        axes[0].set_ylabel('빈도')
        
        # Late Fusion 분포
        late_data = late_fused.detach().cpu().numpy().flatten()
        axes[1].hist(late_data, bins=30, alpha=0.7, color='red')
        axes[1].set_title('Late Fusion 특징 분포')
        axes[1].set_xlabel('특징 값')
        axes[1].set_ylabel('빈도')
        
        plt.tight_layout()
        plt.savefig('fusion_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ 융합 방법 비교 시각화 완료")
    
    def visualize_similarity_matrix(self, similarity_matrix, class_names=None):
        """유사도 행렬 시각화"""
        print("\n📊 유사도 행렬 시각화")
        print("=" * 50)
        
        # 유사도 행렬을 numpy로 변환
        sim_matrix = similarity_matrix.detach().cpu().numpy()
        
        # 히트맵 생성
        plt.figure(figsize=(8, 6))
        sns.heatmap(sim_matrix, annot=True, cmap='viridis', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('이미지-텍스트 유사도 행렬')
        plt.xlabel('텍스트 클래스')
        plt.ylabel('이미지 샘플')
        
        plt.tight_layout()
        plt.savefig('similarity_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ 유사도 행렬 시각화 완료")
    
    def visualize_attention_weights(self, attention_weights):
        """어텐션 가중치 시각화"""
        print("\n📊 어텐션 가중치 시각화")
        print("=" * 50)
        
        # 어텐션 가중치를 numpy로 변환
        attn_weights = attention_weights.detach().cpu().numpy()
        
        # 첫 번째 이미지에 대한 어텐션 가중치
        first_image_attention = attn_weights[0]
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(first_image_attention)), first_image_attention)
        plt.title('첫 번째 이미지의 어텐션 가중치')
        plt.xlabel('텍스트 토큰 인덱스')
        plt.ylabel('어텐션 가중치')
        plt.xticks(range(len(first_image_attention)))
        
        plt.tight_layout()
        plt.savefig('attention_weights.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ 어텐션 가중치 시각화 완료")
    
    def visualize_embedding_space(self, image_embeddings, text_embeddings, class_names):
        """임베딩 공간 시각화 (t-SNE 사용)"""
        print("\n📊 임베딩 공간 시각화")
        print("=" * 50)
        
        try:
            from sklearn.manifold import TSNE
            
            # 임베딩 결합
            combined_embeddings = torch.cat([image_embeddings, text_embeddings], dim=0)
            combined_embeddings = combined_embeddings.detach().cpu().numpy()
            
            # t-SNE로 2D로 차원 축소
            tsne = TSNE(n_components=2, random_state=42)
            embeddings_2d = tsne.fit_transform(combined_embeddings)
            
            # 시각화
            plt.figure(figsize=(10, 8))
            
            # 이미지 임베딩
            n_images = image_embeddings.shape[0]
            plt.scatter(embeddings_2d[:n_images, 0], embeddings_2d[:n_images, 1], 
                       c='blue', marker='o', s=100, alpha=0.7, label='이미지')
            
            # 텍스트 임베딩
            plt.scatter(embeddings_2d[n_images:, 0], embeddings_2d[n_images:, 1], 
                       c='red', marker='s', s=100, alpha=0.7, label='텍스트')
            
            # 클래스 라벨 추가
            for i, class_name in enumerate(class_names):
                plt.annotate(class_name, (embeddings_2d[i, 0], embeddings_2d[i, 1]), 
                           fontsize=8, ha='center')
            
            plt.title('t-SNE 임베딩 공간 시각화')
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('embedding_space.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("✅ 임베딩 공간 시각화 완료")
            
        except ImportError:
            print("⚠️ scikit-learn이 설치되지 않아 t-SNE 시각화를 건너뜁니다.")
    
    def analyze_performance_metrics(self, early_fused, late_fused, similarity_matrix):
        """성능 메트릭 분석"""
        print("\n📊 성능 메트릭 분석")
        print("=" * 50)
        
        # 특징 다양성 분석
        early_diversity = torch.std(early_fused).item()
        late_diversity = torch.std(late_fused).item()
        
        # 유사도 분석
        sim_mean = torch.mean(similarity_matrix).item()
        sim_std = torch.std(similarity_matrix).item()
        sim_max = torch.max(similarity_matrix).item()
        sim_min = torch.min(similarity_matrix).item()
        
        print(f"특징 다양성 (표준편차):")
        print(f"  Early Fusion: {early_diversity:.4f}")
        print(f"  Late Fusion: {late_diversity:.4f}")
        
        print(f"\n유사도 통계:")
        print(f"  평균: {sim_mean:.4f}")
        print(f"  표준편차: {sim_std:.4f}")
        print(f"  최대값: {sim_max:.4f}")
        print(f"  최소값: {sim_min:.4f}")
        
        # 메트릭 시각화
        metrics = {
            'Early Fusion': early_diversity,
            'Late Fusion': late_diversity
        }
        
        plt.figure(figsize=(8, 6))
        plt.bar(metrics.keys(), metrics.values(), color=['blue', 'red'])
        plt.title('융합 방법별 특징 다양성 비교')
        plt.ylabel('표준편차')
        plt.ylim(0, max(metrics.values()) * 1.1)
        
        for i, v in enumerate(metrics.values()):
            plt.text(i, v + 0.001, f'{v:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('performance_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ 성능 메트릭 분석 완료")

def main():
    """메인 실행 함수"""
    print("📊 멀티모달 AI 시각화 및 분석")
    print("=" * 60)
    
    # 시각화 객체 생성
    viz = MultimodalVisualization()
    
    # 더미 데이터 생성
    batch_size = 5
    image_dim = 512
    text_dim = 256
    embedding_dim = 128
    
    image_features = torch.randn(batch_size, image_dim).to(viz.device)
    text_features = torch.randn(batch_size, text_dim).to(viz.device)
    
    class_names = ["car", "pedestrian", "traffic_light", "stop_sign", "construction"]
    
    # 융합 결과 생성
    early_fused = torch.randn(batch_size, 128).to(viz.device)
    late_fused = torch.randn(batch_size, 128).to(viz.device)
    
    # 유사도 행렬 생성
    similarity_matrix = torch.randn(batch_size, batch_size).to(viz.device)
    similarity_matrix = F.softmax(similarity_matrix, dim=-1)
    
    # 어텐션 가중치 생성
    attention_weights = torch.randn(batch_size, batch_size).to(viz.device)
    attention_weights = F.softmax(attention_weights, dim=-1)
    
    # 임베딩 생성
    image_embeddings = torch.randn(batch_size, embedding_dim).to(viz.device)
    text_embeddings = torch.randn(batch_size, embedding_dim).to(viz.device)
    
    # 1. 융합 방법 비교 시각화
    viz.visualize_fusion_comparison(early_fused, late_fused)
    
    # 2. 유사도 행렬 시각화
    viz.visualize_similarity_matrix(similarity_matrix, class_names)
    
    # 3. 어텐션 가중치 시각화
    viz.visualize_attention_weights(attention_weights)
    
    # 4. 임베딩 공간 시각화
    viz.visualize_embedding_space(image_embeddings, text_embeddings, class_names)
    
    # 5. 성능 메트릭 분석
    viz.analyze_performance_metrics(early_fused, late_fused, similarity_matrix)
    
    print("\n✅ 멀티모달 AI 시각화 및 분석 완료!")

if __name__ == "__main__":
    main() 