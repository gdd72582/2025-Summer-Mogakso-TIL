#!/usr/bin/env python3
"""
데이터셋 다운로드 스크립트
KITTI 및 BDD100K 데이터셋의 subset을 다운로드하여 자율주행 객체 탐지 실험에 사용
"""

import os
import sys
import requests
import zipfile
import tarfile
from pathlib import Path
from tqdm import tqdm
import yaml
import argparse
from typing import Dict, List, Optional

# 프로젝트 루트 경로 추가
sys.path.append(str(Path(__file__).parent.parent))

class DatasetDownloader:
    """데이터셋 다운로드 및 전처리 클래스"""
    
    def __init__(self, config_path: str = "../config/data.yaml"):
        """초기화"""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.data_dir = Path(self.config['path'])
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def _load_config(self) -> Dict:
        """설정 파일 로드"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def download_file(self, url: str, filename: str, desc: str = "Downloading") -> Path:
        """파일 다운로드"""
        filepath = self.data_dir / filename
        
        if filepath.exists():
            print(f"파일이 이미 존재합니다: {filepath}")
            return filepath
            
        print(f"다운로드 중: {url}")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        return filepath
    
    def extract_archive(self, filepath: Path, extract_dir: Optional[Path] = None) -> Path:
        """압축 파일 해제"""
        if extract_dir is None:
            extract_dir = filepath.parent
            
        print(f"압축 해제 중: {filepath}")
        
        if filepath.suffix == '.zip':
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
        elif filepath.suffix in ['.tar', '.tar.gz', '.tgz']:
            with tarfile.open(filepath, 'r:*') as tar_ref:
                tar_ref.extractall(extract_dir)
        else:
            raise ValueError(f"지원하지 않는 압축 형식: {filepath.suffix}")
            
        return extract_dir
    
    def download_kitti_subset(self) -> None:
        """KITTI 데이터셋 subset 다운로드"""
        print("=== KITTI 데이터셋 다운로드 ===")
        
        # KITTI 데이터셋 URL (실제로는 더 작은 subset 사용)
        kitti_urls = {
            "images": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip",
            "labels": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip",
            "calib": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip"
        }
        
        kitti_dir = self.data_dir / "kitti_raw"
        kitti_dir.mkdir(exist_ok=True)
        
        for name, url in kitti_urls.items():
            filename = f"kitti_{name}.zip"
            filepath = self.download_file(url, filename, f"KITTI {name}")
            self.extract_archive(filepath, kitti_dir)
            
        print("KITTI 데이터셋 다운로드 완료")
    
    def download_bdd100k_subset(self) -> None:
        """BDD100K 데이터셋 subset 다운로드"""
        print("=== BDD100K 데이터셋 다운로드 ===")
        
        # BDD100K는 더 큰 데이터셋이므로 subset만 다운로드
        # 실제로는 더 작은 샘플 데이터셋 사용
        bdd_urls = {
            "images": "https://bdd-data.berkeley.edu/archive/bdd100k_images_10k.zip",
            "labels": "https://bdd-data.berkeley.edu/archive/bdd100k_labels_release_10k.zip"
        }
        
        bdd_dir = self.data_dir / "bdd100k"
        bdd_dir.mkdir(exist_ok=True)
        
        for name, url in bdd_urls.items():
            filename = f"bdd100k_{name}.zip"
            filepath = self.download_file(url, filename, f"BDD100K {name}")
            self.extract_archive(filepath, bdd_dir)
            
        print("BDD100K 데이터셋 다운로드 완료")
    
    def create_sample_dataset(self) -> None:
        """샘플 데이터셋 생성 (실제 다운로드 대신)"""
        print("=== 샘플 데이터셋 생성 ===")
        
        # 실제 환경에서는 공개 데이터셋을 사용하지만,
        # 여기서는 샘플 구조만 생성
        sample_dir = self.data_dir / "sample_dataset"
        sample_dir.mkdir(exist_ok=True)
        
        # 디렉토리 구조 생성
        for split in ['train', 'val', 'test']:
            (sample_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
            (sample_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
        
        # 샘플 이미지 및 라벨 파일 생성 (실제로는 실제 데이터 사용)
        self._create_sample_files(sample_dir)
        
        print("샘플 데이터셋 생성 완료")
    
    def _create_sample_files(self, sample_dir: Path) -> None:
        """샘플 파일 생성"""
        # 실제로는 실제 이미지와 라벨을 사용해야 함
        # 여기서는 구조만 생성
        
        sample_files = [
            ("train", "000001.jpg"),
            ("train", "000002.jpg"),
            ("val", "000003.jpg"),
            ("test", "000004.jpg")
        ]
        
        for split, filename in sample_files:
            # 샘플 이미지 파일 생성 (실제로는 실제 이미지 복사)
            img_path = sample_dir / 'images' / split / filename
            img_path.touch()  # 빈 파일 생성
            
            # 샘플 라벨 파일 생성
            label_path = sample_dir / 'labels' / split / filename.replace('.jpg', '.txt')
            with open(label_path, 'w') as f:
                # YOLO 형식: class_id x_center y_center width height
                f.write("0 0.5 0.5 0.2 0.3\n")  # car
                f.write("1 0.3 0.7 0.1 0.2\n")  # person
        
        print(f"생성된 샘플 파일: {len(sample_files)}개")
    
    def setup_dataset_structure(self) -> None:
        """데이터셋 구조 설정"""
        print("=== 데이터셋 구조 설정 ===")
        
        # YOLO 형식에 맞는 디렉토리 구조 생성
        for split in ['train', 'val', 'test']:
            (self.data_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
            (self.data_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
        
        # 데이터셋 정보 파일 생성
        dataset_info = {
            'path': str(self.data_dir),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': self.config['nc'],
            'names': self.config['names']
        }
        
        info_path = self.data_dir / 'dataset_info.yaml'
        with open(info_path, 'w', encoding='utf-8') as f:
            yaml.dump(dataset_info, f, default_flow_style=False, allow_unicode=True)
        
        print("데이터셋 구조 설정 완료")
    
    def download_all(self, use_sample: bool = True) -> None:
        """모든 데이터셋 다운로드"""
        print("데이터셋 다운로드 시작...")
        
        if use_sample:
            self.create_sample_dataset()
        else:
            # 실제 데이터셋 다운로드 (시간이 오래 걸림)
            self.download_kitti_subset()
            self.download_bdd100k_subset()
        
        self.setup_dataset_structure()
        print("모든 데이터셋 다운로드 완료!")

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="데이터셋 다운로드 스크립트")
    parser.add_argument('--config', type=str, default='../config/data.yaml',
                       help='설정 파일 경로')
    parser.add_argument('--sample', action='store_true', default=True,
                       help='샘플 데이터셋 사용 (기본값: True)')
    parser.add_argument('--kitti', action='store_true',
                       help='KITTI 데이터셋 다운로드')
    parser.add_argument('--bdd100k', action='store_true',
                       help='BDD100K 데이터셋 다운로드')
    
    args = parser.parse_args()
    
    downloader = DatasetDownloader(args.config)
    
    if args.kitti:
        downloader.download_kitti_subset()
    elif args.bdd100k:
        downloader.download_bdd100k_subset()
    else:
        downloader.download_all(use_sample=args.sample)

if __name__ == "__main__":
    main() 