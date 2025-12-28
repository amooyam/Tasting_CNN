# Tasting_CNN
Implementing layers by myself

Custom CNN Object Detection Project

Overview

This project implements object detection using a custom-built CNN model 
without relying on pre-trained detectors. 
The pipeline includes dataset preparation, training, evaluation, and inference.

Key Features
	Custom CNN backbone and detection head
	Bounding box regression and class prediction
	Train/validation/test split support
	Visualization of detection results

Folder Structure
	configs/ configuration files
	data/ images, labels, dataset splits
	src/ datasets, models, utilities, train/eval/infer scripts
	outputs/ checkpoints, logs, sample results

Requirements

Python 3.9+, PyTorch, torchvision, numpy, OpenCV, matplotlib, PyYAML, tqdm

Dataset

Images and labels are stored under data/.
Train/val/test lists are stored in splits/.
Label format can be YOLO-style text or JSON.

Model
	CNN backbone for feature extraction
	Detection head predicting box coordinates and classes
	Combined loss: localization + objectness + classification

Training

Train with a config file and save checkpoints and logs.
Learning curves and metrics are recorded per epoch.

Evaluation
	mAP, precision, recall
	Confidence and IoU thresholds configurable

Inference

Supports image and optional video inference.
Outputs visualized bounding boxes to outputs/samples/.

Performance and Limitations
	Performance depends on dataset size and model complexity
	May underperform large public models on difficult datasets
	Small or overlapping objects are more challenging

Future Work
	Feature pyramid networks
	Anchor-free detection
	Larger datasets
	Model compression and acceleration


커스텀 CNN 객체 탐지 프로젝트

개요
이 프로젝트는 사전 학습된 모델에 의존하지 않고 직접 설계한 CNN 기반 객체 탐지 모델을 구현한다.
데이터셋 준비, 학습, 평가, 추론을 포함한 전체 파이프라인을 제공한다.

주요 기능
	커스텀 CNN 백본과 디텍션 헤드
	바운딩 박스 회귀 및 클래스 예측
	학습/검증/테스트 데이터 분할 지원
	탐지 결과 시각화

폴더 구조
	configs/ 설정 파일
	data/ 이미지, 라벨, 데이터 분할
	src/ 데이터셋, 모델, 유틸리티, 학습/평가/추론 스크립트
	outputs/ 체크포인트, 로그, 예시 결과

요구 사항

Python 3.9+, PyTorch, torchvision, numpy, OpenCV, matplotlib, PyYAML, tqdm

데이터셋

이미지와 라벨은 data/에 저장된다.
학습/검증/테스트 목록은 splits/에 저장된다.
라벨 형식은 YOLO 텍스트 또는 JSON 형식을 사용할 수 있다.

모델
	특징 추출을 위한 CNN 백본
	박스 좌표와 클래스를 예측하는 디텍션 헤드
	로컬라이제이션 + 오브젝트니스 + 분류를 결합한 손실 함수

학습

설정 파일을 사용해 학습을 수행하며 체크포인트와 로그를 저장한다.
에폭 단위로 학습 곡선과 지표를 기록한다.

평가
	mAP, 정밀도, 재현율
	신뢰도 및 IoU 임계값을 설정할 수 있다

추론

이미지 추론 및 선택적으로 비디오 추론을 지원한다.
바운딩 박스가 시각화된 결과를 outputs/samples/에 저장한다.

성능 및 한계
	성능은 데이터셋 크기와 모델 복잡도에 따라 달라진다
	대형 공개 모델보다 어려운 데이터셋에서는 성능이 낮을 수 있다
	작은 객체와 겹치는 객체 탐지는 더 어렵다

향후 작업
	피처 피라미드 네트워크
	앵커 프리 탐지
	더 큰 데이터셋
	모델 경량화 및 가속
