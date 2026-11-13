"""
=================================================================================
 MicroDefect-AI: Industrial Steel Surface Inspection & Explainability Portal
 Framework: Streamlit | PyTorch | EfficientNet-B0 | Grad-CAM Crack Profiler
=================================================================================
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import cv2
import torch
from torchvision import transforms
from PIL import Image
import json
import base64
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from model import (
    MicroCrackEfficientNetB0,
    MicroGradCAM,
    extract_crack_morphology,
    DEFECT_CLASSES,
    METALLURGICAL_SPECS
)

# ─────────────────────────────────────────────────────────────────────────────
# UI Configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="STEEL VISION — Automated Surface Defect Quality Inspection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enterprise Technical Styling
st.markdown("""
<style>
    .hero-dual-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 14px;
        margin-bottom: 22px;
    }
    .hero-banner-container {
        position: relative;
        width: 100%;
        height: 240px;
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid #334155;
        box-shadow: 0 8px 20px -4px rgba(15, 23, 42, 0.22);
    }
    .hero-banner-img {
        width: 100%;
        height: 100%;
        object-fit: cover;
        filter: contrast(1.05) brightness(0.92);
        transition: transform 0.4s ease, filter 0.4s ease;
    }
    .hero-banner-container:hover .hero-banner-img {
        transform: scale(1.02);
        filter: contrast(1.08) brightness(0.98);
    }
    .hero-banner-overlay {
        position: absolute;
        inset: 0;
        background: linear-gradient(180deg, rgba(15,23,42,0.12) 0%, rgba(15,23,42,0.68) 100%);
        pointer-events: none;
    }
    .hero-banner-tag {
        position: absolute;
        bottom: 12px;
        left: 14px;
        background: rgba(15, 23, 42, 0.78);
        backdrop-filter: blur(6px);
        color: #f1f5f9;
        font-size: 0.74rem;
        font-weight: 600;
        letter-spacing: 0.07em;
        text-transform: uppercase;
        padding: 4px 10px;
        border-radius: 4px;
        border: 1px solid rgba(255, 255, 255, 0.18);
    }
    .enterprise-banner {
        border-bottom: 2px solid #1e293b;
        padding-bottom: 14px;
        margin-bottom: 20px;
    }
    .banner-top-row {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        margin-bottom: 2px;
    }
    .hero-brand {
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, 'Inter', sans-serif;
        font-weight: 900;
        font-size: 2.85rem;
        letter-spacing: -0.03em;
        color: #0f172a;
        line-height: 1.05;
        margin: 0;
        text-transform: uppercase;
    }
    .corner-badge {
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #0f3d64;
        background-color: #f1f5f9;
        border: 1px solid #cbd5e1;
        padding: 5px 12px;
        border-radius: 4px;
    }
    .main-header {
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
        font-weight: 600;
        font-size: 1.35rem;
        color: #334155;
        margin-top: 4px;
        margin-bottom: 4px;
        letter-spacing: -0.01em;
    }
    .sub-header {
        color: #64748b;
        font-size: 0.94rem;
        margin-bottom: 0px;
    }
    .metric-box {
        background: #f8fafc;
        border: 1px solid #cbd5e1;
        border-radius: 6px;
        padding: 12px;
        text-align: center;
    }
    .status-badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }
    .badge-critical { background-color: #fee2e2; color: #991b1b; border: 1px solid #f87171; }
    .badge-high     { background-color: #ffedd5; color: #9a3412; border: 1px solid #fb923c; }
    .badge-moderate { background-color: #fef3c7; color: #92400e; border: 1px solid #facc15; }
    .badge-low      { background-color: #dcfce7; color: #166534; border: 1px solid #4ade80; }
</style>
""", unsafe_allow_html=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────────
# Cached Model Loader
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_system_model(weights_path="best_model.pth"):
    model = MicroCrackEfficientNetB0(num_classes=6, pretrained=False)
    has_weights = False
    if os.path.exists(weights_path):
        try:
            state = torch.load(weights_path, map_location=DEVICE)
            model.load_state_dict(state)
            has_weights = True
            msg = f"Trained weights loaded: {Path(weights_path).name}"
        except Exception as err:
            msg = f"Notice: {err}"
    else:
        msg = "Architecture initialized (Untrained weights)"
    model.to(DEVICE).eval()
    return model, has_weights, msg


def preprocess_image(pil_img):
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return tf(pil_img).unsqueeze(0).to(DEVICE)


def run_micro_inspection(model, input_rgb_224, cam_thresh=0.45, blend_opacity=0.5):
    """
    Performs full deep learning prediction, Grad-CAM activation,
    and morphological contour extraction for quantitative defect profiling.
    """
    pil_img = Image.fromarray(input_rgb_224)
    tensor = preprocess_image(pil_img)

    t0 = time.time()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    inference_time_ms = (time.time() - t0) * 1000

    pred_idx = int(np.argmax(probs))
    pred_class = DEFECT_CLASSES[pred_idx]
    confidence = float(probs[pred_idx])

    # Explainable AI: Grad-CAM
    cam_engine = MicroGradCAM(model)
    cam_map, _ = cam_engine.compute_cam(tensor[0], class_idx=pred_idx)

    # Heatmap colorization
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * cam_map), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    blended = np.uint8(blend_opacity * heatmap_colored + (1.0 - blend_opacity) * input_rgb_224)

    # Morphological defect & crack extraction
    morphology = extract_crack_morphology(cam_map, input_rgb_224, cam_threshold=cam_thresh)

    return {
        "class": pred_class,
        "confidence": confidence,
        "probs": {DEFECT_CLASSES[i]: float(probs[i]) for i in range(len(DEFECT_CLASSES))},
        "latency_ms": inference_time_ms,
        "cam_map": cam_map,
        "heatmap_rgb": heatmap_colored,
        "blended": blended,
        "morphology": morphology,
        "specs": METALLURGICAL_SPECS[pred_class]
    }


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar Controls
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("**Tata Steel Ltd.**")
    st.markdown("*Manufacturing Quality Assurance*")
    st.caption("Surface Inspection Division")
    st.markdown("---")

    st.markdown("### Inspection Configuration")
    ckpt_file = st.text_input("Checkpoint File", value="best_model.pth")
    model, is_loaded, status_msg = load_system_model(ckpt_file)
    if is_loaded:
        st.success(status_msg)
    else:
        st.info(status_msg)

    st.markdown("---")
    st.markdown("#### Morphological Parameters")
    cam_sensitivity = st.slider(
        "Grad-CAM Threshold",
        min_value=0.20,
        max_value=0.80,
        value=0.42,
        step=0.02,
        help="Sensitivity cut-off for isolating micro-crack boundaries"
    )
    overlay_blend = st.slider(
        "Heatmap Opacity",
        min_value=0.1,
        max_value=0.9,
        value=0.45,
        step=0.05
    )

    st.markdown("---")
    st.markdown("#### System Specifications")
    st.markdown(f"- **Backbone**: EfficientNet-B0")
    st.markdown(f"- **Device**: {str(DEVICE).upper()}")
    st.markdown(f"- **Tracked Categories**: {len(DEFECT_CLASSES)}")


# ─────────────────────────────────────────────────────────────────────────────
# Main Screen
# ─────────────────────────────────────────────────────────────────────────────
img1_p = Path("header_steel.jpg")
img2_p = Path("header_steel_line.jpg")

col_img1, col_img2 = st.columns(2)
with col_img1:
    if img1_p.exists():
        st.image(str(img1_p), use_container_width=True)

with col_img2:
    if img2_p.exists():
        st.image(str(img2_p), use_container_width=True)

st.markdown("""
<div class="enterprise-banner">
    <div class="banner-top-row">
        <h1 class="hero-brand">STEEL VISION</h1>
        <div class="corner-badge">Tata Steel Ltd. &bull; Quality Inspection Division</div>
    </div>
    <div class="main-header">Automated Micro-Level Surface Defect & Crack Inspection System</div>
    <div class="sub-header">Automating high-resolution steel plate inspection in real-time manufacturing using EfficientNet-B0 and Morphological Grad-CAM</div>
</div>
""", unsafe_allow_html=True)

nav_tab1, nav_tab2, nav_tab3, nav_tab4 = st.tabs([
    "Microscopic Defect Analysis",
    "NEU-DET Verification Bank",
    "Evaluation Analytics & Graphs",
    "Metallurgical Specifications"
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1: Real-Time Microscopic Defect Analysis
# ═════════════════════════════════════════════════════════════════════════════
with nav_tab1:
    col_input, col_view = st.columns([1, 2], gap="large")

    with col_input:
        st.subheader("Source Micrograph")
        uploaded = st.file_uploader(
            "Upload high-resolution steel plate image",
            type=["jpg", "png", "jpeg", "bmp"]
        )

        if uploaded:
            src_pil = Image.open(uploaded).convert("RGB")
            st.image(src_pil, caption="Optical Sensor Capture", use_container_width=True)
            run_analysis = st.button("Run Defect Inspection", type="primary", use_container_width=True)
        else:
            st.info("Upload a steel plate image to inspect, or select an existing test sample from the NEU-DET Verification Bank.")
            run_analysis = False

    with col_view:
        if uploaded and run_analysis:
            src_arr = np.array(src_pil)
            src_arr = cv2.resize(src_arr, (224, 224))

            with st.spinner("Processing neural inference and Grad-CAM backpropagation..."):
                res = run_micro_inspection(model, src_arr, cam_thresh=cam_sensitivity, blend_opacity=overlay_blend)

            # Quantitative Metrics Row
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Defect Category", res["class"].replace("_", " ").title())
            m2.metric("Confidence", f"{res['confidence']*100:.2f}%")
            m3.metric("Hazard Rating", res["specs"]["hazard_rating"])
            m4.metric("Inference Latency", f"{res['latency_ms']:.1f} ms")

            st.markdown("---")

            # Tri-panel Visuals
            st.markdown("#### Vision & Explainability Views")
            v1, v2, v3 = st.columns(3)
            with v1:
                st.markdown("**1. Raw Microstructure**")
                st.image(src_arr, use_container_width=True)
            with v2:
                st.markdown("**2. Grad-CAM Activation**")
                st.image(res["heatmap_rgb"], use_container_width=True)
            with v3:
                st.markdown("**3. Crack Contour Isolation**")
                st.image(res["morphology"]["annotated_image"], use_container_width=True)

            st.markdown("---")

            # Quantitative defect geometry & Softmax distribution
            q1, q2 = st.columns([1, 1], gap="medium")

            with q1:
                st.markdown("#### Morphological Metrics")
                morph = res["morphology"]
                st.markdown(f"- **Identified Defect Clusters**: `{morph['crack_count']}`")
                st.markdown(f"- **Affected Surface Area**: `{morph['surface_coverage_pct']}%`")
                st.markdown(f"- **Standard Tolerance**: {res['specs']['tolerance']}")

                if len(morph["features"]) > 0:
                    df_features = pd.DataFrame(morph["features"]).rename(columns={
                        "id": "Cluster",
                        "area_px": "Area (px²)",
                        "major_length_px": "Length (px)",
                        "aspect_ratio": "Aspect Ratio",
                        "angle_deg": "Orientation (°)"
                    })
                    st.dataframe(df_features, height=180, use_container_width=True)
                else:
                    st.caption("No continuous crack contours exceeded the sensitivity threshold.")

            with q2:
                st.markdown("#### Class Probability Distribution")
                chart_data = res["probs"]
                fig = go.Figure(go.Bar(
                    x=list(chart_data.values()),
                    y=[c.replace("_", " ").title() for c in DEFECT_CLASSES],
                    orientation="h",
                    marker_color=["#dc2626" if c == res["class"] else "#2563eb" for c in DEFECT_CLASSES],
                    text=[f"{v*100:.1f}%" for v in chart_data.values()],
                    textposition="outside"
                ))
                fig.update_layout(
                    height=240,
                    margin=dict(l=0, r=40, t=10, b=10),
                    xaxis=dict(title="Probability", range=[0, 1.15]),
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2: NEU-DET Verification Bank
# ═════════════════════════════════════════════════════════════════════════════
with nav_tab2:
    st.subheader("Benchmark Validation on NEU Surface Defect Database")

    # Look first in local repo sample_plates (works everywhere including Streamlit Cloud), then fallback to local path
    possible_roots = [
        Path("sample_plates"),
        Path(r"c:\Users\harsh\OneDrive\Desktop\College\AI\tata_steel_inspection\sample_plates"),
        Path(r"C:\Users\harsh\Downloads\NEU-DET\validation\images"),
        Path(r"C:\Users\harsh\Downloads\NEU-DET\train\images"),
        Path(r"c:\Users\harsh\Downloads\archive\NEU-DET\validation\images"),
        Path(r"c:\Users\harsh\Downloads\archive\NEU-DET\train\images"),
    ]

    val_dir = None
    for p in possible_roots:
        if p.exists():
            val_dir = p
            break

    if val_dir is not None:
        sel_cat = st.selectbox(
            "Select Defect Class to Sample:",
            DEFECT_CLASSES,
            format_func=lambda x: f"{x.replace('_', ' ').title()} ({METALLURGICAL_SPECS[x]['hazard_rating']})"
        )

        folder_cat = val_dir / sel_cat
        candidate_files = []
        if folder_cat.exists():
            candidate_files = list(folder_cat.glob("*.jpg")) + list(folder_cat.glob("*.bmp"))

        if candidate_files:
            file_names = [f.name for f in candidate_files[:25]]
            picked_file = st.selectbox("Select Sample Plate:", file_names)
            img_path = folder_cat / picked_file

            c_bench_in, c_bench_out = st.columns([1, 2], gap="large")
            with c_bench_in:
                bench_pil = Image.open(img_path).convert("RGB")
                st.image(bench_pil, caption=f"Sample: {picked_file}", use_container_width=True)
                run_bench = st.button("Inspect Selected Sample", type="primary")

            with c_bench_out:
                if run_bench:
                    arr = np.array(bench_pil)
                    arr = cv2.resize(arr, (224, 224))
                    res_b = run_micro_inspection(model, arr, cam_thresh=cam_sensitivity, blend_opacity=overlay_blend)

                    st.markdown(f"### Result: **{res_b['class'].replace('_', ' ').title()}** ({res_b['confidence']*100:.1f}%)")
                    st.markdown(f"**Tolerance Standard:** {res_b['specs']['tolerance']}")

                    bc1, bc2 = st.columns(2)
                    with bc1:
                        st.image(res_b["heatmap_rgb"], caption="Grad-CAM Focus", use_container_width=True)
                    with bc2:
                        st.image(res_b["morphology"]["annotated_image"], caption="Extracted Contour Map", use_container_width=True)
        else:
            st.warning("No image files found in chosen class directory.")
    else:
        st.info("Upload steel plates using the 'Microscopic Defect Analysis' tab, or include a 'sample_plates/' folder in your repository.")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3: Model Evaluation Analytics & Graphs
# (reads everything from evaluation_results/ — produced by `python evaluate.py`)
# ═════════════════════════════════════════════════════════════════════════════
with nav_tab3:
    st.subheader("Model Evaluation & Diagnostic Analytics")

    eval_dir = Path("evaluation_results")
    rep_json_p = eval_dir / "classification_report.json"
    data_json_p = eval_dir / "eval_data.json"

    def _load_json(p):
        try:
            with open(p, "r") as f:
                return json.load(f)
        except Exception:
            return None

    rep_data = _load_json(rep_json_p) if rep_json_p.exists() else None
    eval_data = _load_json(data_json_p) if data_json_p.exists() else None

    if rep_data is None:
        st.warning(
            "No evaluation results found. Run `python evaluate.py` after training, "
            "then commit the `evaluation_results/` folder (together with `best_model.pth`) "
            "if you deploy to Streamlit Cloud."
        )
    else:
        n_val = int(rep_data["macro avg"]["support"])
        last_run = time.strftime("%d %b %Y, %H:%M", time.localtime(rep_json_p.stat().st_mtime))
        st.markdown(f"Quantitative assessment on the independent validation split ({n_val} steel plate micrographs).")
        st.caption(f"Last evaluated: {last_run}")

        class_labels = [c.replace("_", " ").title() for c in DEFECT_CLASSES]

        # Overview metrics row
        e1, e2, e3, e4, e5 = st.columns(5)
        e1.metric("Validation Accuracy", f"{rep_data['accuracy']*100:.2f}%")
        e2.metric("Macro Avg F1-Score", f"{rep_data['macro avg']['f1-score']:.4f}")
        e3.metric("Weighted Avg Precision", f"{rep_data['weighted avg']['precision']:.4f}")
        e4.metric("Weighted Avg Recall", f"{rep_data['weighted avg']['recall']:.4f}")
        if eval_data:
            e5.metric("Macro ROC-AUC", f"{eval_data['macro_auc']:.4f}")
        else:
            e5.metric("Macro ROC-AUC", "—")

        st.markdown("---")

        # ── 1 & 2: Confusion matrix + ROC ────────────────────────────────────
        g_col1, g_col2 = st.columns(2)
        with g_col1:
            st.markdown("#### 1. Confusion Matrix (Normalized)")
            if eval_data:
                cm_norm = np.array(eval_data["confusion_matrix_normalized"])
                cm_cnt = np.array(eval_data["confusion_matrix_counts"])
                fig_cm = go.Figure(data=go.Heatmap(
                    z=cm_norm, x=class_labels, y=class_labels,
                    colorscale="Blues", zmin=0, zmax=1,
                    text=[[f"{cm_norm[i][j]*100:.1f}%<br>({cm_cnt[i][j]})" for j in range(len(class_labels))]
                          for i in range(len(class_labels))],
                    texttemplate="%{text}", showscale=True
                ))
                fig_cm.update_layout(
                    xaxis_title="Predicted Class", yaxis_title="True Class",
                    yaxis=dict(autorange="reversed"),
                    height=420, margin=dict(l=0, r=20, t=10, b=10)
                )
                st.plotly_chart(fig_cm, use_container_width=True)
            elif (eval_dir / "confusion_matrix.png").exists():
                st.image(str(eval_dir / "confusion_matrix.png"), use_container_width=True)
            else:
                st.info("Re-run `python evaluate.py` to generate this chart.")

        with g_col2:
            st.markdown("#### 2. ROC-AUC Performance Curves")
            if eval_data:
                palette = ["#2563eb", "#d97706", "#ea580c", "#dc2626", "#0d9488", "#16a34a"]
                fig_roc = go.Figure()
                for i, name in enumerate(class_labels):
                    r = eval_data["roc"][name]
                    fig_roc.add_trace(go.Scatter(
                        x=r["fpr"], y=r["tpr"], mode="lines",
                        line=dict(color=palette[i % len(palette)], width=2.2),
                        name=f"{name} (AUC = {r['auc']:.3f})"
                    ))
                fig_roc.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1], mode="lines",
                    line=dict(dash="dash", color="#64748b", width=1.5), name="Random (0.500)"
                ))
                fig_roc.update_layout(
                    xaxis_title="False Positive Rate (FPR)",
                    yaxis_title="True Positive Rate (Sensitivity / Recall)",
                    height=420, margin=dict(l=0, r=20, t=10, b=10),
                    legend=dict(font=dict(size=9.5), yanchor="bottom", y=0.03, xanchor="right", x=0.98)
                )
                st.plotly_chart(fig_roc, use_container_width=True)
            elif (eval_dir / "roc_curves.png").exists():
                st.image(str(eval_dir / "roc_curves.png"), use_container_width=True)
            else:
                st.info("Re-run `python evaluate.py` to generate this chart.")

        st.markdown("---")

        # ── 3 & 4: Per-class bars + report table ─────────────────────────────
        g_col3, g_col4 = st.columns([1.2, 0.8])
        with g_col3:
            st.markdown("#### 3. Precision, Recall & F1-Score Breakdown")
            fig_bar = go.Figure(data=[
                go.Bar(name="Precision", x=class_labels, y=[rep_data[c]["precision"] for c in class_labels], marker_color="#3b82f6"),
                go.Bar(name="Recall", x=class_labels, y=[rep_data[c]["recall"] for c in class_labels], marker_color="#10b981"),
                go.Bar(name="F1-Score", x=class_labels, y=[rep_data[c]["f1-score"] for c in class_labels], marker_color="#6366f1"),
            ])
            fig_bar.update_layout(
                barmode="group", yaxis_title="Score (0.0 - 1.0)", yaxis=dict(range=[0, 1.1]),
                height=340, margin=dict(l=0, r=10, t=10, b=10)
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with g_col4:
            st.markdown("#### 4. Classification Report Table")
            table_rows = [{
                "Class": c,
                "Precision": round(rep_data[c]["precision"], 3),
                "Recall": round(rep_data[c]["recall"], 3),
                "F1": round(rep_data[c]["f1-score"], 3),
                "Support": int(rep_data[c]["support"]),
            } for c in class_labels if c in rep_data]
            st.dataframe(pd.DataFrame(table_rows), height=260, use_container_width=True)

        # ── 5: Training curves ───────────────────────────────────────────────
        hist = eval_data.get("history") if eval_data else None
        if hist:
            st.markdown("---")
            st.markdown("#### 5. Training vs Validation Curves")
            h1, h2 = st.columns(2)
            with h1:
                fig_l = go.Figure()
                fig_l.add_trace(go.Scatter(x=hist["epoch"], y=hist["train_loss"], mode="lines+markers", name="Train Loss", line=dict(color="#ef4444")))
                fig_l.add_trace(go.Scatter(x=hist["epoch"], y=hist["val_loss"], mode="lines+markers", name="Val Loss", line=dict(color="#3b82f6")))
                fig_l.update_layout(xaxis_title="Epoch", yaxis_title="Cross-Entropy Loss", height=300, margin=dict(l=0, r=10, t=10, b=10))
                st.plotly_chart(fig_l, use_container_width=True)
            with h2:
                fig_a = go.Figure()
                fig_a.add_trace(go.Scatter(x=hist["epoch"], y=[a * 100 for a in hist["train_acc"]], mode="lines+markers", name="Train Accuracy", line=dict(color="#10b981")))
                fig_a.add_trace(go.Scatter(x=hist["epoch"], y=[a * 100 for a in hist["val_acc"]], mode="lines+markers", name="Val Accuracy", line=dict(color="#6366f1")))
                fig_a.update_layout(xaxis_title="Epoch", yaxis_title="Accuracy (%)", height=300, margin=dict(l=0, r=10, t=10, b=10))
                st.plotly_chart(fig_a, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4: Metallurgical Specifications
# ═════════════════════════════════════════════════════════════════════════════
with nav_tab4:
    st.subheader("Defect Engineering Knowledge Base")
    
    spec_rows = []
    for cls_name, details in METALLURGICAL_SPECS.items():
        spec_rows.append({
            "Defect Category": cls_name.replace("_", " ").title(),
            "Hazard Rating": details["hazard_rating"],
            "Description": details["description"],
            "Factory Tolerance": details["tolerance"]
        })

    st.table(pd.DataFrame(spec_rows))

    st.markdown("""
### Technical Principles
- **Backbone Network**: EfficientNet-B0 with ImageNet transfer learning fine-tuned on metallurgical microstructure.
- **Micro-Crack Localization**: Grad-CAM captures spatial feature activations from the terminal MBConv block.
- **Morphological Profiling**: Adaptive contour extraction and ellipse fitting measure orientation angle (theta), major axis length, and surface coverage percentage.
""")
