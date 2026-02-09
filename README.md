# DRIFT

**DRIFT**: Robust Multi-Domain Digital Pathology Image Segmentation via Joint Balancing Representation Learning.

This repository provides the research implementation of **DRIFT**, a robust multi-domain digital pathology image segmentation framework built upon **nnUNetv2**.  
The proposed method introduces joint balancing representation learning to improve cross-domain generalization under significant domain shifts in histopathology images.

---

## Overview

Deep learning models for digital pathology often suffer from severe performance degradation when deployed across domains due to variations in staining protocols, scanners, and tissue characteristics.  
DRIFT addresses this challenge by integrating a **joint balancing representation learning strategy** into the nnUNetv2 training pipeline, enabling more stable and robust multi-domain segmentation.

This repository provides:

- A customized nnUNetv2 training pipeline with DRIFT-specific modifications  
- A lightweight implementation of the PRISM-DD related module  
- Full compatibility with standard `nnUNetv2_train` commands  

---

## Repository Structure

```text
DRIFT/
├── s2s3/                          # Main nnUNetv2-based codebase with DRIFT modifications
│   ├── training/
│   │   └── nnUNetTrainer/         # Customized nnUNetTrainer for DRIFT
│   ├── run/
│   ├── utilities/
│   └── ...
├── s1/
│   └── PRISMDD.py                 # PRISM-DD related module
├── pyproject.toml                 # Package configuration (maps s2s3 → nnunetv2)
├── requirements.txt
├── .gitignore
└── README.md
``` 
---
## Installation

### 1. Environment

We recommend using **conda** to manage the runtime environment.

```bash
conda create -n nnunetv2 python=3.9
conda activate nnunetv2
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install -e .
nnUNetv2_train -h
```

### 2. Usage

## Usage

DRIFT follows the standard **nnUNetv2** command-line interface.

### Training

```bash
nnUNetv2_train <dataset_name_or_id> <configuration> <fold> [options]

```
### 3. Data Preparation


Please follow the official **nnUNetv2** data preparation guidelines:

https://github.com/MIC-DKFZ/nnUNet

> **Important**  
> Due to data privacy and licensing restrictions, no datasets or trained models are included in this repository.
### 4. Implementation Notes

- DRIFT modifies the **nnUNetTrainer** to incorporate joint balancing representation learning.  
- The original nnUNetv2 training workflow and configuration system are preserved.  
- The design ensures minimal disruption to nnUNetv2 while enabling robust multi-domain learning.

### 5. Reproducibility

- The code is designed to be fully reproducible using standard nnUNetv2 pipelines.  
- Random seeds and training configurations follow nnUNetv2 defaults unless explicitly modified.
### 6. Citation

If you find this work useful, please cite our paper:

```bibtex
@article{DRIFT2024,
  title   = {Robust Multi-Domain Digital Pathology Image Segmentation via Joint Balancing Representation Learning},
  author  = {Author Names},
  journal = {Journal Name},
  year    = {2024}
}
```
## Acknowledgements

This work is built upon the excellent **nnUNetv2** framework:

> Isensee et al., *nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation*.

We thank the authors for making their code publicly available.

## License

This repository is released for academic research purposes.  
Please refer to the original **nnUNetv2** license for third-party components.





