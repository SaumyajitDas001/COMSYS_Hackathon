
# Gender Classification from Images

A simple deep learning pipeline for binary gender classification using either:

- ✅ CNN from scratch
- ✅ MobileNetV2 (transfer learning)

## 🔧 Setup

```bash
pip install -r requirements.txt
```

## 📁 Dataset Structure

```
dataset/
├── train/
│   ├── male/
│   └── female/
└── val/
    ├── male/
    └── female/
```

## 🧠 Train the Model

```bash
python train.py
```

## 🎯 Evaluate on a Single Image

```bash
python evaluate.py
```

The image will be shown and deleted automatically after prediction.

## 🧩 Model Architecture

Choose model type in `train.py` via:
- `"cnn"` for a custom model
- `"mobilenetv2"` for transfer learning
- `"both"` to use MobileNetV2 when possible, fallback to CNN if offline
