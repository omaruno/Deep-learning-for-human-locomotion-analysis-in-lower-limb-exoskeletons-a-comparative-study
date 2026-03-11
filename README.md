# Deep Learning for Human Locomotion Analysis in Lower-Limb Exoskeletons

Official implementation of the paper:

> **Deep Learning for Human Locomotion Analysis in Lower-Limb Exoskeletons: A Comparative Study**
>
> O. Coser, C. Tamantini, M. Tortora, L. Furia, R. Sicilia, L. Zollo, P. Soda
>
> *Frontiers in Computer Science*, 2025 — [DOI: 10.3389/fcomp.2025.1597143](https://doi.org/10.3389/fcomp.2025.1597143)

---

## Overview

This repository provides a deep learning framework for real-time locomotion analysis in lower-limb exoskeletons, addressing two tasks:

**Terrain classification** (5 classes):
- Level Ground (LG)
- Ramp Ascent (RA)
- Ramp Descent (RD)
- Stair Ascent (SA)
- Stair Descent (SD)

**Locomotion parameter estimation:**
- Ramp slope regression
- Stair height regression

### Key Results

| Task | Metric | Value |
|------|--------|-------|
| Terrain classification | Accuracy | 0.94 ± 0.04 |
| Ramp slope regression | MAE | ~1.95° |
| Stair height regression | MAE | ~15.65 mm |
| Inference time | Latency | ~1–2 ms |

---

## Implemented Architectures

We provide implementations and comparisons of eight state-of-the-art time-series deep learning models:

| Model | Type |
|-------|------|
| CNN | Convolutional |
| LSTM | Recurrent |
| GRU | Recurrent |
| CNN-LSTM | Hybrid |
| LSTM-CNN | Hybrid |
| XceptionTime | Convolutional (depthwise separable) |
| Time Series Transformer (TST) | Attention-based |
| Mamba | State Space Model |

### Best Performing Models

| Task | Best Model |
|------|------------|
| Terrain classification | LSTM |
| Ramp slope regression | LSTM |
| Stair height regression | CNN-LSTM |

---

## Dataset

Experiments are conducted using the **CAMARGO 2021** dataset:

- 21 subjects
- 4 IMUs (trunk, thigh, shank, foot)
- 11 EMG sensors
- Multiple ramp inclinations
- Multiple stair heights

The study demonstrates that an **IMU-only setup (foot, shank, thigh)** matches or outperforms IMU+EMG configurations, enabling a lightweight and cost-efficient wearable system.

---

## Explainable AI (SHAP)

This repository includes SHAP-based feature importance analysis to identify the most informative sensors and determine a minimal sensor configuration.

**Key findings:**
- **Foot IMU** (gyroscope Y-axis) is the most informative feature
- **Trunk IMU** can be removed without performance loss
- **Three IMUs** (foot, shank, thigh) are sufficient for full performance

---

## Experimental Setup

- **Validation:** Leave-One-Subject-Out (LOSO) cross-validation
- **Window size:** 100 ms non-overlapping windows
- **Preprocessing:** Standardized across all subjects
- **Training:** Early stopping to prevent overfitting
- **Statistical testing:** Wilcoxon signed-rank test with Bonferroni correction

---

## Applications

- Adaptive control for lower-limb exoskeletons
- Assistive and rehabilitation robotics
- Terrain-aware locomotion systems
- Gait analysis research

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{coser2025deep,
  title={Deep learning for human locomotion analysis in lower-limb exoskeletons: a comparative study},
  author={Coser, Omar and Tamantini, Christian and Tortora, Matteo and Furia, Leonardo and Sicilia, Rosa and Zollo, Loredana and Soda, Paolo},
  journal={Frontiers in Computer Science},
  volume={7},
  pages={1597143},
  year={2025},
  publisher={Frontiers Media SA}
}
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

