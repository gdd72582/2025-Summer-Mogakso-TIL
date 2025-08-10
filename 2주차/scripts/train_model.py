#!/usr/bin/env python3
"""
YOLO 모델 학습 스크립트
설정 파일을 기반으로 YOLO 모델을 학습하고 실험 결과를 저장
"""

import os
import sys
import yaml
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# 프로젝트 루트 경로 추가
sys.path.append(str(Path(__file__).parent.parent))

# YOLO 관련 import
try:
    from ultralytics import YOLO
except ImportError:
    print("ultralytics 패키지가 설치되지 않았습니다. pip install ultralytics를 실행하세요.")
    sys.exit(1)

class YOLOTrainer:
    """YOLO 모델 학습 클래스"""
    
    def __init__(self, config_path: str, exp_name: Optional[str] = None):
        """초기화"""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.exp_name = exp_name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 로깅 설정
        self._setup_logging()
        
        # 실험 디렉토리 설정
        self.exp_dir = Path("experiments") / self.exp_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # 설정 파일 복사
        self._save_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """설정 파일 로드"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _setup_logging(self):
        """로깅 설정"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'experiments/{self.exp_name}/training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _save_config(self):
        """설정 파일을 실험 디렉토리에 저장"""
        config_save_path = self.exp_dir / "config.yaml"
        with open(config_save_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
        self.logger.info(f"설정 파일 저장: {config_save_path}")
    
    def _prepare_training_args(self, **kwargs) -> Dict[str, Any]:
        """학습 인자 준비"""
        training_config = self.config.get('training', {})
        augmentation_config = self.config.get('augmentation', {})
        
        # 기본 학습 인자
        args = {
            'data': 'config/data.yaml',
            'model': training_config.get('model', 'yolov8s.pt'),
            'epochs': training_config.get('epochs', 100),
            'batch': training_config.get('batch_size', 16),
            'imgsz': training_config.get('imgsz', 640),
            'device': self.config.get('optimization', {}).get('device', 'auto'),
            'project': 'experiments',
            'name': self.exp_name,
            'save_period': training_config.get('save_period', 10),
            'cache': self.config.get('optimization', {}).get('cache', False),
            'workers': self.config.get('optimization', {}).get('workers', 8),
            'amp': self.config.get('optimization', {}).get('amp', True),
        }
        
        # 학습률 관련
        lr_config = training_config.get('lr0', 0.01)
        if isinstance(lr_config, dict):
            args.update(lr_config)
        else:
            args['lr0'] = lr_config
            args['lrf'] = training_config.get('lrf', 0.1)
        
        # 옵티마이저 설정
        optimizer = training_config.get('optimizer', 'SGD')
        if optimizer == 'SGD':
            args['momentum'] = training_config.get('momentum', 0.937)
            args['weight_decay'] = training_config.get('weight_decay', 0.0005)
            args['nesterov'] = training_config.get('nesterov', True)
        
        # 데이터 증강 설정
        for key, value in augmentation_config.items():
            if key in ['hsv_h', 'hsv_s', 'hsv_v']:
                args[f'hsv_{key.split("_")[-1]}'] = value
            else:
                args[key] = value
        
        # 커맨드 라인 인자로 덮어쓰기
        args.update(kwargs)
        
        return args
    
    def train(self, **kwargs) -> str:
        """모델 학습 실행"""
        self.logger.info(f"학습 시작: {self.exp_name}")
        
        # 학습 인자 준비
        train_args = self._prepare_training_args(**kwargs)
        
        # YOLO 모델 초기화
        model = YOLO(train_args['model'])
        
        # 학습 실행
        try:
            results = model.train(**train_args)
            self.logger.info("학습 완료!")
            
            # 결과 저장
            self._save_results(results)
            
            return str(results.save_dir)
            
        except Exception as e:
            self.logger.error(f"학습 중 오류 발생: {e}")
            raise
    
    def _save_results(self, results):
        """학습 결과 저장"""
        # 결과 요약 저장
        results_summary = {
            'experiment_name': self.exp_name,
            'training_time': datetime.now().isoformat(),
            'best_map': results.results_dict.get('metrics/mAP50(B)', 0),
            'best_map_50_95': results.results_dict.get('metrics/mAP50-95(B)', 0),
            'final_loss': results.results_dict.get('train/box_loss', 0),
        }
        
        summary_path = self.exp_dir / "results_summary.yaml"
        with open(summary_path, 'w', encoding='utf-8') as f:
            yaml.dump(results_summary, f, default_flow_style=False, allow_unicode=True)
        
        self.logger.info(f"결과 요약 저장: {summary_path}")
    
    def validate(self, model_path: str) -> Dict[str, float]:
        """모델 검증"""
        self.logger.info("모델 검증 시작")
        
        model = YOLO(model_path)
        
        # 검증 설정
        val_config = self.config.get('validation', {})
        val_args = {
            'data': 'config/data.yaml',
            'conf': val_config.get('conf', 0.001),
            'iou': val_config.get('iou', 0.6),
            'max_det': val_config.get('max_det', 300),
            'half': val_config.get('half', True),
            'save_json': val_config.get('save_json', False),
            'save_hybrid': val_config.get('save_hybrid', False),
        }
        
        # 검증 실행
        results = model.val(**val_args)
        
        # 결과 저장
        val_results = {
            'mAP50': results.box.map50,
            'mAP50-95': results.box.map,
            'precision': results.box.mp,
            'recall': results.box.mr,
        }
        
        val_path = self.exp_dir / "validation_results.yaml"
        with open(val_path, 'w', encoding='utf-8') as f:
            yaml.dump(val_results, f, default_flow_style=False, allow_unicode=True)
        
        self.logger.info(f"검증 결과 저장: {val_path}")
        return val_results

def run_ablation_study(config_path: str, ablation_config: Dict[str, Any]):
    """Ablation 실험 실행"""
    print("=== Ablation 실험 시작 ===")
    
    ablation_studies = ablation_config.get('ablation_studies', {})
    
    for study_name, experiments in ablation_studies.items():
        print(f"\n--- {study_name} 실험 ---")
        
        for i, exp_config in enumerate(experiments):
            exp_name = exp_config.get('name', f"{study_name}_{i}")
            print(f"실험: {exp_name}")
            
            # 실험별 설정 업데이트
            trainer = YOLOTrainer(config_path, exp_name)
            
            # 실험별 파라미터로 학습
            trainer.train(**exp_config)
            
            print(f"실험 {exp_name} 완료")

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="YOLO 모델 학습 스크립트")
    parser.add_argument('--config', type=str, required=True,
                       help='설정 파일 경로')
    parser.add_argument('--exp_name', type=str,
                       help='실험 이름')
    parser.add_argument('--model', type=str,
                       help='모델 파일 경로 (기본값: 설정 파일에서 읽음)')
    parser.add_argument('--epochs', type=int,
                       help='학습 에포크 수')
    parser.add_argument('--batch', type=int,
                       help='배치 크기')
    parser.add_argument('--imgsz', type=int,
                       help='입력 이미지 크기')
    parser.add_argument('--lr0', type=float,
                       help='초기 학습률')
    parser.add_argument('--ablation', action='store_true',
                       help='Ablation 실험 실행')
    parser.add_argument('--validate', type=str,
                       help='검증할 모델 경로')
    
    args = parser.parse_args()
    
    # 커맨드 라인 인자 준비
    train_kwargs = {}
    if args.model:
        train_kwargs['model'] = args.model
    if args.epochs:
        train_kwargs['epochs'] = args.epochs
    if args.batch:
        train_kwargs['batch'] = args.batch
    if args.imgsz:
        train_kwargs['imgsz'] = args.imgsz
    if args.lr0:
        train_kwargs['lr0'] = args.lr0
    
    # 설정 파일 로드
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    if args.ablation:
        # Ablation 실험 실행
        run_ablation_study(args.config, config)
    elif args.validate:
        # 모델 검증
        trainer = YOLOTrainer(args.config, "validation")
        results = trainer.validate(args.validate)
        print("검증 결과:", results)
    else:
        # 일반 학습
        trainer = YOLOTrainer(args.config, args.exp_name)
        results_dir = trainer.train(**train_kwargs)
        print(f"학습 완료! 결과 저장 위치: {results_dir}")

if __name__ == "__main__":
    main() 