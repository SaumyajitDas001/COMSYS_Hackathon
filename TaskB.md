
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
- Accuracy
- Precision (macro)
- Recall (macro)
- F1 Score (macro)
- Per-class classification report
