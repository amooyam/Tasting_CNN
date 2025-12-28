# Tasting_CNN
Implementing layers by myself

Custom CNN Object Detection Project

Overview

This project implements object detection using a custom-built CNN model 
without relying on pre-trained detectors. 
The pipeline includes dataset preparation, training, evaluation, and inference.

Key Features
	•	Custom CNN backbone and detection head
	•	Bounding box regression and class prediction
	•	Train/validation/test split support
	•	Visualization of detection results

Folder Structure
	•	configs/ configuration files
	•	data/ images, labels, dataset splits
	•	src/ datasets, models, utilities, train/eval/infer scripts
	•	outputs/ checkpoints, logs, sample results

Requirements

Python 3.9+, PyTorch, torchvision, numpy, OpenCV, matplotlib, PyYAML, tqdm

Dataset

Images and labels are stored under data/.
Train/val/test lists are stored in splits/.
Label format can be YOLO-style text or JSON.

Model
	•	CNN backbone for feature extraction
	•	Detection head predicting box coordinates and classes
	•	Combined loss: localization + objectness + classification

Training

Train with a config file and save checkpoints and logs.
Learning curves and metrics are recorded per epoch.

Evaluation
	•	mAP, precision, recall
	•	Confidence and IoU thresholds configurable

Inference

Supports image and optional video inference.
Outputs visualized bounding boxes to outputs/samples/.

Performance and Limitations
	•	Performance depends on dataset size and model complexity
	•	May underperform large public models on difficult datasets
	•	Small or overlapping objects are more challenging

Future Work
	•	Feature pyramid networks
	•	Anchor-free detection
	•	Larger datasets
	•	Model compression and acceleration
