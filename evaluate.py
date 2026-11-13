"""
=================================================================================
 Tata Steel Ltd. — Model Evaluation & Diagnostic Report Generator
 Computes Confusion Matrix, Precision, Recall, F1-Score, ROC-AUC, 
 and Training/Validation Loss-Accuracy Curves.
=================================================================================
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from model import MicroCrackEfficientNetB0, DEFECT_CLASSES
from train import SteelDataset, build_pipeline_transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_system(data_dir, checkpoint_path="best_model.pth", output_dir="evaluation_results"):
    os.makedirs(output_dir, exist_ok=True)
    print(f"[Device] Running evaluation on: {DEVICE}")

    # 1. Load Dataset
    _, val_tf = build_pipeline_transforms()
    val_set = SteelDataset(data_dir, split="validation", transform=val_tf)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
    print(f"[Dataset] Found {len(val_set)} validation images across {len(DEFECT_CLASSES)} classes.")
    if len(val_set) == 0:
        raise FileNotFoundError(
            f"No validation images found under '{data_dir}'. "
            "Check --data_dir (it must contain a 'validation' folder)."
        )

    # 2. Load Model
    model = MicroCrackEfficientNetB0(num_classes=6, pretrained=False)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
    
    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE).eval()
    print(f"[Model] Successfully loaded weights from {checkpoint_path}")

    # 3. Inference Loop
    y_true = []
    y_pred = []
    y_probs = []

    print("[Inference] Gathering predictions on validation set...")
    with torch.no_grad():
        for imgs, labels, _ in val_loader:
            imgs = imgs.to(DEVICE)
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            y_true.extend(labels.numpy())
            y_pred.extend(preds)
            y_probs.extend(probs)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)

    class_names_clean = [c.replace("_", " ").title() for c in DEFECT_CLASSES]

    # 4. Generate Classification Report
    report_dict = classification_report(y_true, y_pred, target_names=class_names_clean, output_dict=True)
    report_text = classification_report(y_true, y_pred, target_names=class_names_clean, digits=4)
    print("\n" + "="*65)
    print("TATA STEEL AUTOMATED QUALITY INSPECTION — EVALUATION REPORT")
    print("="*65)
    print(report_text)

    with open(os.path.join(output_dir, "classification_report.json"), "w") as f:
        json.dump(report_dict, f, indent=2)

    # -------------------------------------------------------------
    # GRAPH 1: Confusion Matrix Heatmap
    # -------------------------------------------------------------
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(cm_normalized, annot=True, fmt=".2%", cmap="Blues",
                xticklabels=class_names_clean, yticklabels=class_names_clean,
                cbar_kws={'label': 'Normalized Accuracy'}, ax=ax)
    ax.set_title("Tata Steel Inspection AI — Normalized Confusion Matrix", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Defect Category", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Ground Truth Category", fontsize=11, fontweight="bold")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[Graph Saved] Confusion Matrix -> {cm_path}")

    # -------------------------------------------------------------
    # GRAPH 2: Multi-Class ROC-AUC Curves
    # -------------------------------------------------------------
    y_true_bin = label_binarize(y_true, classes=list(range(len(DEFECT_CLASSES))))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for i in range(len(DEFECT_CLASSES)):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        ax.plot(fpr[i], tpr[i], color=colors[i], lw=2,
                label=f"{class_names_clean[i]} (AUC = {roc_auc[i]:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Guess (AUC = 0.500)")
    ax.set_xlim([-0.02, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (FPR)", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Positive Rate (TPR / Sensitivity)", fontsize=11, fontweight="bold")
    ax.set_title("Receiver Operating Characteristic (ROC) per Defect Class", fontsize=12, fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    roc_path = os.path.join(output_dir, "roc_curves.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"[Graph Saved] ROC Curves -> {roc_path}")

    # -------------------------------------------------------------
    # GRAPH 3: Per-Class F1, Precision & Recall Comparison Bar Graph
    # -------------------------------------------------------------
    metrics_names = ["Precision", "Recall", "F1-Score"]
    precisions = [report_dict[c]["precision"] for c in class_names_clean]
    recalls = [report_dict[c]["recall"] for c in class_names_clean]
    f1s = [report_dict[c]["f1-score"] for c in class_names_clean]

    x = np.arange(len(class_names_clean))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - width, precisions, width, label='Precision', color="#3b82f6")
    ax.bar(x, recalls, width, label='Recall', color="#10b981")
    ax.bar(x + width, f1s, width, label='F1-Score', color="#6366f1")

    ax.set_ylabel('Score (0.0 - 1.0)', fontsize=11, fontweight="bold")
    ax.set_title('Precision, Recall, and F1-Score by Defect Class', fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names_clean, rotation=20, ha="right")
    ax.set_ylim([0.0, 1.15])
    ax.legend(loc="upper right")
    ax.grid(axis='y', alpha=0.3)

    for i in range(len(class_names_clean)):
        ax.text(x[i] - width, precisions[i] + 0.02, f"{precisions[i]:.2f}", ha='center', fontsize=8)
        ax.text(x[i], recalls[i] + 0.02, f"{recalls[i]:.2f}", ha='center', fontsize=8)
        ax.text(x[i] + width, f1s[i] + 0.02, f"{f1s[i]:.2f}", ha='center', fontsize=8)

    plt.tight_layout()
    bar_path = os.path.join(output_dir, "class_performance_metrics.png")
    plt.savefig(bar_path, dpi=300)
    plt.close()
    print(f"[Graph Saved] Class Metrics Bar Graph -> {bar_path}")

    # -------------------------------------------------------------
    # GRAPH 4: Training & Validation Curves (if training_log.json exists)
    # -------------------------------------------------------------
    hist = None
    for cand in ("training_log.json", "training_history.json"):
        if Path(cand).exists():
            with open(cand, "r") as f:
                hist = json.load(f)
            break
    if hist is not None:

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
        epochs_range = hist.get("epoch", list(range(1, len(hist["train_loss"]) + 1)))

        # Loss curve
        ax1.plot(epochs_range, hist["train_loss"], "o-", label="Train Loss", color="#ef4444")
        ax1.plot(epochs_range, hist["val_loss"], "s-", label="Val Loss", color="#3b82f6")
        ax1.set_title("Training vs Validation Loss", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Epoch", fontsize=10)
        ax1.set_ylabel("Cross-Entropy Loss", fontsize=10)
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Accuracy curve
        ax2.plot(epochs_range, [acc * 100 for acc in hist["train_acc"]], "o-", label="Train Accuracy", color="#10b981")
        ax2.plot(epochs_range, [acc * 100 for acc in hist["val_acc"]], "s-", label="Val Accuracy", color="#6366f1")
        ax2.set_title("Training vs Validation Accuracy", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Epoch", fontsize=10)
        ax2.set_ylabel("Accuracy (%)", fontsize=10)
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        curves_path = os.path.join(output_dir, "training_validation_curves.png")
        plt.savefig(curves_path, dpi=300)
        plt.close()
        print(f"[Graph Saved] Learning Curves -> {curves_path}")


    # -------------------------------------------------------------
    # Raw evaluation data for the Streamlit app (interactive charts)
    # -------------------------------------------------------------
    n_hist = len(hist["train_loss"]) if hist else 0
    eval_data = {
        "class_names": class_names_clean,
        "accuracy": float(report_dict["accuracy"]),
        "n_samples": int(len(y_true)),
        "confusion_matrix_counts": cm.tolist(),
        "confusion_matrix_normalized": cm_normalized.tolist(),
        "roc": {
            class_names_clean[i]: {
                "fpr": [round(float(v), 4) for v in fpr[i]],
                "tpr": [round(float(v), 4) for v in tpr[i]],
                "auc": round(float(roc_auc[i]), 4),
            } for i in range(len(DEFECT_CLASSES))
        },
        "macro_auc": round(float(np.mean(list(roc_auc.values()))), 4),
        "history": {
            "epoch": hist.get("epoch", list(range(1, n_hist + 1))),
            "train_loss": hist["train_loss"], "val_loss": hist["val_loss"],
            "train_acc": hist["train_acc"], "val_acc": hist["val_acc"],
        } if hist else None,
    }
    with open(os.path.join(output_dir, "eval_data.json"), "w") as f:
        json.dump(eval_data, f)
    print(f"[Data Saved] App chart data -> {os.path.join(output_dir, 'eval_data.json')}")

    print("\n[Done] All evaluation graphs and reports generated successfully in:", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=r"C:\Users\harsh\Downloads\NEU-DET")
    parser.add_argument("--checkpoint", type=str, default="best_model.pth")
    parser.add_argument("--out_dir", type=str, default="evaluation_results")
    args = parser.parse_args()

    evaluate_system(args.data_dir, args.checkpoint, args.out_dir)
