# DRIFT

**DRIFT**: Robust Multi-Domain Digital Pathology Image Segmentation via Joint Balancing Representation Learning.

This repository provides the official research code for **DRIFT**, a robust multi-domain digital pathology image segmentation framework built upon **nnUNetv2**.  
The implementation introduces joint balancing representation learning to improve cross-domain generalization in histopathology image segmentation.

---

## Overview

Deep learning models for digital pathology often suffer from significant performance degradation when deployed across domains due to variations in staining protocols, scanners, and tissue characteristics.  
DRIFT addresses this challenge by integrating a **joint balancing representation learning strategy** into the nnUNetv2 training pipeline, enabling more robust and stable multi-domain segmentation.

This repository contains:
- A customized nnUNetv2 training pipeline with DRIFT-specific modifications.
- A lightweight implementation of the PRISM-DD module.
- Full compatibility with standard `nnUNetv2_train` commands.

---

## Repository Structure

```text
DRIFT/
├── s2s3/                          # Main nnUNetv2-based codebase with DRIFT modifications
│   ├── training/
│   │   └── nnUNetTrainer/          # Customized nnUNetTrainer for DRIFT
│   ├── run/
│   ├── utilities/
│   └── ...
├── s1/
│   └── PRISMDD.py                  # PRISM-DD related module
├── pyproject.toml                  # Package configuration (maps s2s3 → nnunetv2)
├── .gitignore
└── README.md
