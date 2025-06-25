
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from model import FaceNet, Classifier
from dataset import DistortedDataset
import argparse
import os

def evaluate(val_path, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    class_to_idx = checkpoint['class_to_idx']
    num_classes = len(class_to_idx)

    model = FaceNet().to(device)
    classifier = Classifier(128, num_classes).to(device)
    model.load_state_dict(checkpoint['embedding_model'])
    classifier.load_state_dict(checkpoint['classifier'])

    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    val_dataset = DistortedDataset(val_path, class_to_idx, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model.eval()
    classifier.eval()

    preds, trues = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            emb = model(imgs)
            out = classifier(emb)
            pred = torch.argmax(out, dim=1)
            preds.append(pred.item())
            trues.append(labels.item())

    if len(preds) > 0:
        print("✅ Accuracy:", accuracy_score(trues, preds))
        print("📊 Precision (macro):", precision_score(trues, preds, average='macro', zero_division=0))
        print("📊 Recall (macro):", recall_score(trues, preds, average='macro', zero_division=0))
        print("📊 F1 Score (macro):", f1_score(trues, preds, average='macro', zero_division=0))

        idx_to_class = {v: k for k, v in class_to_idx.items()}
        used_labels = sorted(set(trues + preds))
        used_names = [idx_to_class[i] for i in used_labels]
        print("\n📋 Classification Report:")
        print(classification_report(trues, preds, labels=used_labels, target_names=used_names, zero_division=0))
    else:
        print("🚫 No predictions made.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="models/face_model.pth")
    args = parser.parse_args()
    evaluate(args.val_path, args.model_path)
