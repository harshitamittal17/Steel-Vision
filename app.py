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
    neu_path = Path(r"c:\Users\harsh\Downloads\archive\NEU-DET")

    val_dir = neu_path / "validation" / "images"
    if not val_dir.exists():
        val_dir = neu_path / "train" / "images"

    if val_dir.exists():
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
        st.warning(f"NEU-DET database not found at `{neu_path}`.")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3: Model Evaluation Analytics & Graphs
# ═════════════════════════════════════════════════════════════════════════════
with nav_tab3:
    st.subheader("Model Evaluation & Diagnostic Analytics")
    st.markdown("Quantitative assessment on the independent validation split (360 high-resolution steel plate micrographs).")

    eval_dir = Path("evaluation_results")
    cm_img_p = eval_dir / "confusion_matrix.png"
    roc_img_p = eval_dir / "roc_curves.png"
    bar_img_p = eval_dir / "class_performance_metrics.png"
    rep_json_p = eval_dir / "classification_report.json"

    if rep_json_p.exists():
        with open(rep_json_p, "r") as f:
            rep_data = json.load(f)

        # Overview Metrics Row
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Validation Accuracy", f"{rep_data['accuracy']*100:.2f}%")
        e2.metric("Macro Avg F1-Score", f"{rep_data['macro avg']['f1-score']:.4f}")
        e3.metric("Weighted Avg Precision", f"{rep_data['weighted avg']['precision']:.4f}")
        e4.metric("Weighted Avg Recall", f"{rep_data['weighted avg']['recall']:.4f}")

        st.markdown("---")

        # Visual Graphs 2x2
        g_col1, g_col2 = st.columns(2)
        with g_col1:
            st.markdown("#### 1. Confusion Matrix (Normalized)")
            if cm_img_p.exists():
                st.image(str(cm_img_p), use_container_width=True)
        with g_col2:
            st.markdown("#### 2. ROC-AUC Performance Curves")
            if roc_img_p.exists():
                st.image(str(roc_img_p), use_container_width=True)

        st.markdown("---")

        g_col3, g_col4 = st.columns([1.2, 0.8])
        with g_col3:
            st.markdown("#### 3. Precision, Recall & F1-Score Breakdown")
            if bar_img_p.exists():
                st.image(str(bar_img_p), use_container_width=True)
        with g_col4:
            st.markdown("#### 4. Classification Report Table")
            table_rows = []
            for cls_k in [c.replace("_", " ").title() for c in DEFECT_CLASSES]:
                if cls_k in rep_data:
                    table_rows.append({
                        "Class": cls_k,
                        "Precision": round(rep_data[cls_k]["precision"], 3),
                        "Recall": round(rep_data[cls_k]["recall"], 3),
                        "F1": round(rep_data[cls_k]["f1-score"], 3),
                        "Support": int(rep_data[cls_k]["support"])
                    })
            st.dataframe(pd.DataFrame(table_rows), height=240, use_container_width=True)
    else:
        st.info("Run `python evaluate.py` to compile evaluation metrics and plots.")


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
