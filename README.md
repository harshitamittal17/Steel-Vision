# STEEL VISION — Automated Surface Defect Quality Inspection System

**STEEL VISION** is a high-performance deep convolutional vision platform developed for **Tata Steel Ltd.** to automate micro-level surface crack and defect detection in manufacturing processes from high-resolution optical imagery.

---

## Technical Overview
In continuous steel plate manufacturing, early identification of micro-level surface cracks and imperfections is critical to preventing downstream structural failures. This automated quality inspection system integrates:
- **EfficientNet-B0 Backbone** with Transfer Learning
- **Multi-Angle Industrial Data Augmentation** for factory floor robustness
- **Explainable AI (Grad-CAM)** with **Morphological Crack Boundary Extraction** to quantify defect length, orientation angle, and surface coverage percentage
- **Empirical Validation** against the **NEU Surface Defect Database (NEU-DET)**

---

## Defect Categories (NEU-DET)
1. **Crazing (Critical)**: Interconnected micro-crack networks caused by thermal fatigue or high tensile stress.
2. **Inclusion (High)**: Trapped non-metallic impurities (slag/oxides) weakening yield strength.
3. **Patches (Moderate)**: Localized thickness and texture variations from uneven rolling reduction.
4. **Pitted Surface (Critical)**: Acid wash cavities and corrosion micro-pits acting as mechanical stress risers.
5. **Rolled-in Scale (High)**: Dense iron oxide mill scale embedded in strip surface during hot rolling.
6. **Scratches (Low)**: Tooling abrasion marks; generally superficial.

---

## Explainable AI & Quantitative Crack Profiling
Standard classification models output only a single category label. This system couples Grad-CAM gradient maps with computer vision morphology:
- **Contour Extraction**: Segments localized micro-crack boundaries.
- **Ellipse Fitting**: Computes major crack length ($L$), aspect ratio ($L/W$), and orientation ($\theta$).
- **Surface Coverage**: Measures the total percentage of plate area affected.

---

## Execution Guide

### 1. Install Dependencies
```bash
pip install torch torchvision opencv-python streamlit plotly pandas pillow
```

### 2. Train the EfficientNet-B0 Model
```bash
python train.py --data_dir "c:\Users\harsh\Downloads\archive\NEU-DET" --epochs 5 --batch_size 16 --save_path "best_model.pth"
```

### 3. Launch the Inspection Portal
```bash
streamlit run app.py
```
