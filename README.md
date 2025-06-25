
# Task B: Face Recognition - Multi-Class Classification

This repository contains the implementation for a multi-class face recognition system that classifies distorted images.

## 📁 Folder Structure
- `train/PersonName/*.jpg` - Clean images
- `val/PersonName/distortion/*.jpg` - Distorted images

## 📦 Files
- `model.py` - FaceNet and classifier architecture
- `dataset.py` - Custom PyTorch datasets for clean and distorted images
- `evaluate.py` - Evaluates the model using Accuracy, Precision, Recall, F1
- `models/face_model.pth` - Pretrained model weights (you must add this)
- `README.md` - Instructions and overview

## 🧪 How to Evaluate
```bash
python evaluate.py --val_path path/to/TaskB/val
```

## 📝 Metrics Returned
Accuracy: 0.0030
Precision (Macro): 0.0039
Recall (Macro): 0.0023
F1 Score (Macro): 0.0023
Precision (Weighted): 0.0048
Recall (Weighted): 0.0030
F1 Score (Weighted): 0.0029
