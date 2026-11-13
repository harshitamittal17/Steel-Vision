"""
=================================================================================
 MicroDefect-AI: Precision Training & Validation Harness (v2)
 EfficientNet-B0 (ImageNet-pretrained) + Industrial Augmentation on NEU-DET

 Changes vs v1:
   * ImageNet-pretrained backbone (was training from scratch)
   * Discriminative learning rates (backbone lower, new head higher)
   * Linear warm-up + cosine decay
   * More epochs (30) with early stopping
   * Mixed precision (AMP) on GPU, gradient clipping
   * Affine augmentation (small shift/scale) + mild random erasing
   * Best checkpoint chosen by val accuracy, ties broken by val loss
=================================================================================
"""

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
    """Returns (train_tf, val_tf). Signature unchanged so evaluate.py keeps working."""
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomAffine(degrees=20, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.25, contrast=0.25),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.08), value=0),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return train_tf, val_tf


def run_training(dataset_path, epochs=30, batch_size=32, lr=1e-3, backbone_lr_mult=0.3,
                 warmup_epochs=2, patience=8, checkpoint_out="best_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"[Device] Using {device} (AMP: {use_amp})")

    train_tf, val_tf = build_pipeline_transforms()
    train_set = SteelDataset(dataset_path, split="train", transform=train_tf)
    val_set = SteelDataset(dataset_path, split="validation", transform=val_tf)

    print(f"[Dataset] Train samples: {len(train_set)}, Validation samples: {len(val_set)}")
    if len(train_set) == 0:
        raise FileNotFoundError(f"No steel images found in {dataset_path}")

    pin = device.type == "cuda"
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=pin, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=pin)

    # Pretrained backbone is the single biggest accuracy lever.
    model = MicroCrackEfficientNetB0(num_classes=6, pretrained=True).to(device)

    head_params = list(model.backbone.classifier.parameters())
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": lr * backbone_lr_mult},
        {"params": head_params, "lr": lr},
    ], weight_decay=1e-2)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs - warmup_epochs, 1), eta_min=1e-6)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_acc, best_loss, bad_epochs = 0.0, float("inf"), 0
    history = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print("\n--- Starting Training Run (EfficientNet-B0, pretrained) ---")
    for epoch in range(1, epochs + 1):
        start_t = time.time()
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for imgs, labels, _ in train_loader:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(imgs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * imgs.size(0)
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_total += labels.size(0)

        scheduler.step()
        train_loss /= train_total
        train_acc = train_correct / train_total

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels, _ in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    logits = model(imgs)
                    loss = criterion(logits, labels)
                val_loss += loss.item() * imgs.size(0)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)
        elapsed = time.time() - start_t

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch:02d}/{epochs:02d} [{elapsed:.1f}s] - "
              f"Loss: {train_loss:.4f} Acc: {train_acc*100:.1f}% | "
              f"Val Loss: {val_loss:.4f} Val Acc: {val_acc*100:.2f}%")

        improved = val_acc > best_acc or (val_acc == best_acc and val_loss < best_loss)
        if improved:
            best_acc, best_loss, bad_epochs = val_acc, val_loss, 0
            torch.save(model.state_dict(), checkpoint_out)
            print(f" -> Saved new best checkpoint ({best_acc*100:.2f}%) to {checkpoint_out}")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"[Early stop] No improvement for {patience} epochs.")
                break

    with open("training_log.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n[Done] Best validation accuracy: {best_acc*100:.2f}%")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=r"c:\Users\harsh\Downloads\archive\NEU-DET")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3, help="Head LR; backbone uses lr * 0.3")
    parser.add_argument("--save_path", type=str, default="best_model.pth")
    args = parser.parse_args()

    run_training(args.data_dir, args.epochs, args.batch_size, args.lr, checkpoint_out=args.save_path)
