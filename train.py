"""
=================================================================================
 MicroDefect-AI: Precision Training & Validation Harness
 EfficientNet-B0 + Industrial Data Augmentation on NEU Surface Defect Database
=================================================================================
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from model import MicroCrackEfficientNetB0, DEFECT_CLASSES


class SteelDataset(Dataset):
    def __init__(self, dataset_root, split="train", transform=None):
        self.transform = transform
        self.samples = []

        target_dir = Path(dataset_root) / split
        img_base = target_dir / "images" if (target_dir / "images").exists() else target_dir

        for label_idx, class_name in enumerate(DEFECT_CLASSES):
            folder = img_base / class_name
            if folder.exists() and folder.is_dir():
                for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                    for img_p in folder.glob(ext):
                        self.samples.append((str(img_p), label_idx))
            else:
                for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                    for img_p in img_base.glob(f"{class_name}_*{ext[1:]}"):
                        self.samples.append((str(img_p), label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p, label = self.samples[idx]
        img = Image.open(p).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label, p


def build_pipeline_transforms():
    """
    Industrial augmentation pipeline tailored to micro-crack orientation and sensor variation.
    """
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.25, contrast=0.25),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_tf, val_tf


def run_training(dataset_path, epochs=5, batch_size=16, lr=2e-4, checkpoint_out="best_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using {device} for model execution.")

    train_tf, val_tf = build_pipeline_transforms()
    train_set = SteelDataset(dataset_path, split="train", transform=train_tf)
    val_set = SteelDataset(dataset_path, split="validation", transform=val_tf)

    print(f"[Dataset] Train samples: {len(train_set)}, Validation samples: {len(val_set)}")
    if len(train_set) == 0:
        raise FileNotFoundError(f"No steel images found in {dataset_path}")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = MicroCrackEfficientNetB0(num_classes=6, pretrained=False).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.08)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_accuracy = 0.0
    history = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print("\n--- Starting Training Run (EfficientNet-B0) ---")
    for epoch in range(1, epochs + 1):
        start_t = time.time()
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for imgs, labels, _ in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            train_correct += (preds.argmax(1) == labels).sum().item()
            train_total += labels.size(0)

        scheduler.step()
        train_loss /= train_total
        train_acc = train_correct / train_total

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels, _ in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs)
                loss = criterion(preds, labels)
                val_loss += loss.item() * imgs.size(0)
                val_correct += (preds.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_loss / max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)
        elapsed = time.time() - start_t

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch:02d}/{epochs:02d} [{elapsed:.1f}s] - "
              f"Loss: {train_loss:.4f} Acc: {train_acc*100:.1f}% | "
              f"Val Loss: {val_loss:.4f} Val Acc: {val_acc*100:.1f}%")

        if val_acc > best_accuracy or epoch == 1:
            best_accuracy = val_acc
            torch.save(model.state_dict(), checkpoint_out)
            print(f" -> Saved top performing model checkpoint to {checkpoint_out}")

    with open("training_log.json", "w") as f:
        json.dump(history, f, indent=2)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=r"c:\Users\harsh\Downloads\archive\NEU-DET")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--save_path", type=str, default="best_model.pth")
    args = parser.parse_args()

    run_training(args.data_dir, args.epochs, args.batch_size, args.lr, args.save_path)
