#!/usr/bin/env python3
"""
모델 성능 벤치마크 스크립트
YOLO 모델의 FPS, 지연시간, 메모리 사용량 등을 측정
"""

import os
import sys
import time
import argparse
import yaml
import json
import numpy as np
import psutil
import GPUtil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
import torch

# 프로젝트 루트 경로 추가
sys.path.append(str(Path(__file__).parent.parent))

try:
    from ultralytics import YOLO
except ImportError:
    print("ultralytics 패키지가 설치되지 않았습니다.")
    sys.exit(1)

class ModelBenchmark:
    """모델 성능 벤치마크 클래스"""
    
    def __init__(self, model_path: str, config_path: Optional[str] = None):
        """초기화"""
        self.model_path = Path(model_path)
        self.config = self._load_config(config_path) if config_path else {}
        
        # 모델 로드
        self.model = YOLO(str(self.model_path))
        
        # 하드웨어 정보 수집
        self.hardware_info = self._get_hardware_info()
        
        # 결과 저장
        self.results = {}
        
    def _load_config(self, config_path: str) -> Dict:
        """설정 파일 로드"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _get_hardware_info(self) -> Dict:
        """하드웨어 정보 수집"""
        info = {
            'cpu': {
                'count': psutil.cpu_count(),
                'frequency': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {},
                'memory': psutil.virtual_memory()._asdict()
            },
            'gpu': {}
        }
        
        # GPU 정보 수집
        try:
            gpus = GPUtil.getGPUs()
            for i, gpu in enumerate(gpus):
                info['gpu'][f'gpu_{i}'] = {
                    'name': gpu.name,
                    'memory_total': gpu.memoryTotal,
                    'memory_free': gpu.memoryFree,
                    'temperature': gpu.temperature,
                    'load': gpu.load
                }
        except Exception as e:
            print(f"GPU 정보 수집 실패: {e}")
        
        return info
    
    def _generate_test_images(self, img_size: int, num_images: int = 100) -> List[np.ndarray]:
        """테스트용 이미지 생성"""
        images = []
        for i in range(num_images):
            # 랜덤 이미지 생성
            img = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)
            images.append(img)
        return images
    
    def _load_test_images(self, image_dir: str, img_size: int) -> List[np.ndarray]:
        """테스트 이미지 로드"""
        images = []
        image_path = Path(image_dir)
        
        if not image_path.exists():
            print(f"이미지 디렉토리가 존재하지 않습니다: {image_dir}")
            return self._generate_test_images(img_size)
        
        # 지원하는 이미지 확장자
        extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        for ext in extensions:
            for img_file in image_path.glob(f"*{ext}"):
                try:
                    img = cv2.imread(str(img_file))
                    if img is not None:
                        img = cv2.resize(img, (img_size, img_size))
                        images.append(img)
                except Exception as e:
                    print(f"이미지 로드 실패 {img_file}: {e}")
        
        if not images:
            print("로드된 이미지가 없습니다. 랜덤 이미지를 생성합니다.")
            return self._generate_test_images(img_size)
        
        return images
    
    def _measure_memory_usage(self) -> Dict:
        """메모리 사용량 측정"""
        memory_info = {}
        
        # CPU 메모리
        cpu_memory = psutil.virtual_memory()
        memory_info['cpu'] = {
            'total': cpu_memory.total,
            'available': cpu_memory.available,
            'used': cpu_memory.used,
            'percent': cpu_memory.percent
        }
        
        # GPU 메모리
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_stats()
            memory_info['gpu'] = {
                'allocated': torch.cuda.memory_allocated(),
                'cached': torch.cuda.memory_reserved(),
                'max_allocated': torch.cuda.max_memory_allocated()
            }
        
        return memory_info
    
    def benchmark_inference(self, 
                          img_size: int = 640,
                          num_iterations: int = 1000,
                          warmup_iterations: int = 100,
                          batch_size: int = 1,
                          image_dir: Optional[str] = None) -> Dict:
        """추론 성능 벤치마크"""
        print(f"=== 추론 성능 벤치마크 시작 ===")
        print(f"이미지 크기: {img_size}x{img_size}")
        print(f"배치 크기: {batch_size}")
        print(f"워밍업 반복: {warmup_iterations}")
        print(f"벤치마크 반복: {num_iterations}")
        
        # 테스트 이미지 준비
        if image_dir:
            test_images = self._load_test_images(image_dir, img_size)
        else:
            test_images = self._generate_test_images(img_size, max(num_iterations, 100))
        
        # 추론 설정
        inference_config = self.config.get('inference', {})
        conf_thres = inference_config.get('conf_thres', 0.25)
        iou_thres = inference_config.get('iou_thres', 0.6)
        
        # 워밍업
        print("워밍업 중...")
        for _ in range(warmup_iterations):
            img = test_images[0]
            _ = self.model(img, conf=conf_thres, iou=iou_thres, verbose=False)
        
        # GPU 동기화
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # 메모리 사용량 측정 (시작)
        memory_before = self._measure_memory_usage()
        
        # 벤치마크 시작
        print("벤치마크 실행 중...")
        latencies = []
        fps_measurements = []
        
        start_time = time.time()
        
        for i in range(num_iterations):
            img = test_images[i % len(test_images)]
            
            # 추론 시간 측정
            inference_start = time.time()
            results = self.model(img, conf=conf_thres, iou=iou_thres, verbose=False)
            inference_end = time.time()
            
            latency = (inference_end - inference_start) * 1000  # ms
            latencies.append(latency)
            
            # FPS 계산 (실시간)
            if i > 0:
                elapsed_time = time.time() - start_time
                current_fps = i / elapsed_time
                fps_measurements.append(current_fps)
        
        total_time = time.time() - start_time
        
        # 메모리 사용량 측정 (종료)
        memory_after = self._measure_memory_usage()
        
        # 결과 계산
        avg_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        min_latency = np.min(latencies)
        max_latency = np.max(latencies)
        
        avg_fps = num_iterations / total_time
        avg_fps_real_time = np.mean(fps_measurements) if fps_measurements else avg_fps
        
        # 메모리 사용량 변화
        memory_usage = {
            'cpu_used': memory_after['cpu']['used'] - memory_before['cpu']['used'],
            'cpu_percent': memory_after['cpu']['percent']
        }
        
        if 'gpu' in memory_after:
            memory_usage['gpu_allocated'] = memory_after['gpu']['allocated']
            memory_usage['gpu_cached'] = memory_after['gpu']['cached']
        
        # 결과 저장
        benchmark_results = {
            'model_path': str(self.model_path),
            'image_size': img_size,
            'batch_size': batch_size,
            'num_iterations': num_iterations,
            'warmup_iterations': warmup_iterations,
            
            # 성능 지표
            'avg_latency_ms': avg_latency,
            'std_latency_ms': std_latency,
            'min_latency_ms': min_latency,
            'max_latency_ms': max_latency,
            'avg_fps': avg_fps,
            'avg_fps_real_time': avg_fps_real_time,
            'total_time_seconds': total_time,
            
            # 메모리 사용량
            'memory_usage': memory_usage,
            
            # 하드웨어 정보
            'hardware_info': self.hardware_info,
            
            # 추론 설정
            'inference_config': {
                'conf_thres': conf_thres,
                'iou_thres': iou_thres
            }
        }
        
        self.results['inference'] = benchmark_results
        
        # 결과 출력
        print(f"\n=== 벤치마크 결과 ===")
        print(f"평균 지연시간: {avg_latency:.2f} ms")
        print(f"지연시간 표준편차: {std_latency:.2f} ms")
        print(f"최소 지연시간: {min_latency:.2f} ms")
        print(f"최대 지연시간: {max_latency:.2f} ms")
        print(f"평균 FPS: {avg_fps:.2f}")
        print(f"실시간 평균 FPS: {avg_fps_real_time:.2f}")
        print(f"총 실행 시간: {total_time:.2f} 초")
        
        return benchmark_results
    
    def benchmark_different_sizes(self, sizes: List[int] = [320, 640, 800, 1024]) -> Dict:
        """다양한 이미지 크기에서 벤치마크"""
        print("=== 다양한 이미지 크기 벤치마크 ===")
        
        size_results = {}
        
        for size in sizes:
            print(f"\n이미지 크기 {size}x{size} 테스트 중...")
            results = self.benchmark_inference(img_size=size, num_iterations=500)
            size_results[f"{size}x{size}"] = results
        
        self.results['size_comparison'] = size_results
        
        # 결과 요약
        print(f"\n=== 크기별 성능 비교 ===")
        for size, result in size_results.items():
            print(f"{size}: {result['avg_fps']:.2f} FPS, {result['avg_latency_ms']:.2f} ms")
        
        return size_results
    
    def export_results(self, output_path: str):
        """결과를 파일로 저장"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # JSON 형식으로 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"벤치마크 결과 저장: {output_path}")
    
    def generate_report(self, output_path: str):
        """벤치마크 리포트 생성"""
        if not self.results:
            print("벤치마크 결과가 없습니다. 먼저 벤치마크를 실행하세요.")
            return
        
        report = []
        report.append("# YOLO 모델 벤치마크 리포트")
        report.append("")
        
        # 하드웨어 정보
        report.append("## 하드웨어 정보")
        report.append(f"- CPU: {self.hardware_info['cpu']['count']} 코어")
        if self.hardware_info['gpu']:
            for gpu_id, gpu_info in self.hardware_info['gpu'].items():
                report.append(f"- GPU {gpu_id}: {gpu_info['name']}")
        report.append("")
        
        # 추론 성능
        if 'inference' in self.results:
            inference = self.results['inference']
            report.append("## 추론 성능")
            report.append(f"- 모델: {inference['model_path']}")
            report.append(f"- 이미지 크기: {inference['image_size']}x{inference['image_size']}")
            report.append(f"- 평균 FPS: {inference['avg_fps']:.2f}")
            report.append(f"- 평균 지연시간: {inference['avg_latency_ms']:.2f} ms")
            report.append("")
        
        # 크기별 비교
        if 'size_comparison' in self.results:
            report.append("## 크기별 성능 비교")
            report.append("| 이미지 크기 | FPS | 지연시간 (ms) |")
            report.append("|-------------|-----|---------------|")
            
            for size, result in self.results['size_comparison'].items():
                fps = result['avg_fps']
                latency = result['avg_latency_ms']
                report.append(f"| {size} | {fps:.2f} | {latency:.2f} |")
            report.append("")
        
        # 메모리 사용량
        if 'inference' in self.results:
            memory = self.results['inference']['memory_usage']
            report.append("## 메모리 사용량")
            report.append(f"- CPU 메모리 사용률: {memory['cpu_percent']:.1f}%")
            if 'gpu_allocated' in memory:
                gpu_mb = memory['gpu_allocated'] / (1024 * 1024)
                report.append(f"- GPU 메모리 할당: {gpu_mb:.1f} MB")
            report.append("")
        
        # 리포트 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print(f"벤치마크 리포트 저장: {output_path}")

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="YOLO 모델 벤치마크 스크립트")
    parser.add_argument('--model', type=str, required=True,
                       help='모델 파일 경로')
    parser.add_argument('--config', type=str,
                       help='설정 파일 경로')
    parser.add_argument('--img_size', type=int, default=640,
                       help='테스트 이미지 크기')
    parser.add_argument('--iterations', type=int, default=1000,
                       help='벤치마크 반복 횟수')
    parser.add_argument('--warmup', type=int, default=100,
                       help='워밍업 반복 횟수')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='배치 크기')
    parser.add_argument('--image_dir', type=str,
                       help='테스트 이미지 디렉토리')
    parser.add_argument('--sizes', nargs='+', type=int, default=[320, 640, 800],
                       help='테스트할 이미지 크기들')
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                       help='결과 저장 파일 경로')
    parser.add_argument('--report', type=str, default='benchmark_report.md',
                       help='리포트 저장 파일 경로')
    parser.add_argument('--size_benchmark', action='store_true',
                       help='다양한 이미지 크기에서 벤치마크 실행')
    
    args = parser.parse_args()
    
    # 벤치마크 실행
    benchmark = ModelBenchmark(args.model, args.config)
    
    if args.size_benchmark:
        # 다양한 크기에서 벤치마크
        benchmark.benchmark_different_sizes(args.sizes)
    else:
        # 단일 크기에서 벤치마크
        benchmark.benchmark_inference(
            img_size=args.img_size,
            num_iterations=args.iterations,
            warmup_iterations=args.warmup,
            batch_size=args.batch_size,
            image_dir=args.image_dir
        )
    
    # 결과 저장
    benchmark.export_results(args.output)
    benchmark.generate_report(args.report)

if __name__ == "__main__":
    main() 