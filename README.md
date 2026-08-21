# Knee Cartilage Pipeline

Deep learning pipeline for automatic 3D knee cartilage segmentation, atlas-based registration, and morphological biomarker extraction from MRI data. Developed as part of an internship at ICube Laboratory, University of Strasbourg (INSERM Regenerative Nanomedicine).

## Clinical Context

Osteochondral damage in the knee joint often leads to chronic osteoarthritis, a degenerative condition with no existing cure. The Regenerative Nanomedicine INSERM lab has developed a biodegradable biomimetic implant to support cartilage and bone regeneration, currently under clinical trials. This pipeline serves as a preparatory study for monitoring treatment efficacy through quantitative MRI analysis.

## Pipeline Overview

```
MRI Input (SKM-TEA)
        ↓
  Preprocessing (1mm isotropic)
        ↓
  Segmentation
  NCA v5 vs UNet3D
  4 classes: Patellar, Femoral, Tibial, Meniscus
        ↓
  Atlas-based Registration
  Template from 85 train subjects (itk-elastix)
        ↓
  Morphological Biomarkers
  Volume · Thickness · Surface Area
```

## Dataset

| Dataset | Subjects | Scanner | Sequence | Split |
|---------|----------|---------|----------|-------|
| Stanford SKM-TEA | 154 | GE 3T | qDESS (2-echo) | 85 train / 33 val / 36 test |

- **SKM-TEA**: Available at [Stanford AIMI](https://aimi.stanford.edu/skm-tea)

## Results

### Segmentation (Test Set, n=36)

| Model | DSC | HD95 (mm) | VS | Params |
|-------|-----|-----------|-----|--------|
| NCA v5 (ours) | 0.7908 | 2.03 | 0.9117 | ~30K |
| UNet3D | 0.8309 | 3.22 | 0.9099 | 23.5M |

Per-class DSC (NCA v5):

| Structure | DSC | HD95 (mm) |
|-----------|-----|-----------|
| Patellar | 0.7539 | 2.36 |
| Femoral | 0.8076 | 2.00 |
| Tibial | 0.8223 | 1.58 |
| Meniscus | 0.7797 | 2.19 |

### Morphological Biomarkers (Subject Space, n=36)

| Structure | Volume (cm³) | Thickness (mm) | Surface Area (mm²) |
|-----------|-------------|----------------|-------------------|
| Patellar | 3.48 ± 0.87 | 2.51 ± 0.15 | 2185 ± 393 |
| Femoral | 13.56 ± 2.92 | 2.26 ± 0.07 | 10042 ± 1661 |
| Tibial | 5.08 ± 1.08 | 2.28 ± 0.12 | 3972 ± 638 |
| Meniscus | 4.25 ± 1.12 | 2.52 ± 0.11 | 2730 ± 561 |

Thickness values consistent with published literature. Coefficient of variation for thickness: 3–6%.

## Project Structure

```
knee-cartilage-pipeline/
├── README.md
├── segmentation/
│   ├── m3dnca/
│   │   ├── train_mednca_v5_4class.py     # NCA v5 training
│   │   └── test_mednca_v5_4class.py      # NCA v5 inference + evaluation
│   └── unet3d/
│       ├── train_unet3d_4class.py        # UNet3D training
│       └── test_unet3d_4class.py         # UNet3D inference + evaluation
├── registration/
│   ├── build_template_full.py            # Atlas construction (85 subjects)
│   ├── register_to_template_2.py         # Subject-to-template registration
│   └── morphometrics_subject_space.py    # Morphological biomarker extraction
├── visualization/
│   └── visualize_morphology_v2.py        # Morphology plots
└── results/
    ├── morphology_summary.csv            # Per-subject morphology
    ├── morphology_stats.json             # Mean ± std per structure
    └── viz/                              # Visualization figures
        ├── 1_boxplot_morphology.png
        ├── 2_scatter_vol_vs_thick.png
        ├── 3_heatmap_per_subject.png
        └── 4_bar_mean_std.png
```

## How to Run

### 1. Segmentation

Train and evaluate segmentation models on SKM-TEA dataset.

```bash
# NCA v5
python segmentation/m3dnca/train_mednca_v5_4class.py
python segmentation/m3dnca/test_mednca_v5_4class.py

# UNet3D
python segmentation/unet3d/train_unet3d_4class.py
python segmentation/unet3d/test_unet3d_4class.py
```

### 2. Registration + Morphology

Build atlas template, register test subjects, and extract morphological biomarkers.

```bash
# Step 1: Build atlas template from all 85 train subjects
python registration/build_template_full.py

# Step 2: Register 36 test subjects to template
python registration/register_to_template_2.py

# Or: compute morphology directly in subject space (faster)
python registration/morphometrics_subject_space.py
```

### 3. Visualization

Generate morphology plots.

```bash
python visualization/visualize_morphology_v2.py
```

## Installation

```bash
pip install torch numpy SimpleITK itk-elastix scipy pandas matplotlib scikit-image tqdm
```

## Resources

| Resource | Link |
|----------|------|
| SKM-TEA preprocessed (1mm isotropic) | [https://drive.google.com/drive/folders/146JFbZ9s0-1ZV0fEg2KiGvDILvIYpVRE?usp=drive_link](#) |
| M3DNCA segmentation results (NIfTI for 3D Slicer) | [https://drive.google.com/drive/folders/1Ot5Dz6VNdEmebEjR2KCOnR-yncNmlfWG?usp=sharing](#) |
| Model checkpoints | [https://drive.google.com/drive/folders/1If_HH1bJkxYOzhKACF4Mj4vry-57DJrw?usp=drive_link](#) |

> **Note:** Update paths in each script according to your local setup before running.

## References

- Kalkhof et al., Med-NCA: Robust Medical Image Segmentation using Neural Cellular Automata, IPMI 2023
- Yao et al., CartiMorph: A framework for automated knee articular cartilage morphometrics, Medical Image Analysis 2024. DOI: 10.1016/j.media.2023.103035

## Acknowledgements

This work was supervised by Prof. Caroline Essert, Dr. Nadia Benkirane-Jessel, and Dr. Rana Smaida at ICube Laboratory, University of Strasbourg. Conducted in collaboration with the INSERM Regenerative Nanomedicine team.
