# Skin Cancer Detection with Federated Learning

A notebook-based deep learning project for skin lesion classification using a pretrained Swin Transformer model and a federated learning setup. The repository focuses on detecting multiple classes of skin cancer from medical images and includes data validation, model training, evaluation, and Grad-CAM visual explanations.

## Overview

This project aims to build an automated skin cancer detection system that can classify skin lesions into multiple categories. The implementation uses:

- Swin Transformer (Swin-Tiny) pretrained on ImageNet
- Federated learning with a FedDC-inspired update strategy
- Image preprocessing and data loaders
- Model evaluation using accuracy and macro F1-score
- Grad-CAM for visual interpretability

The project is primarily implemented as Jupyter notebooks and is suitable for experimentation, research, and learning.

## Key Features

- Skin lesion classification using deep learning
- Pretrained Swin Transformer backbone
- Federated learning approach for distributed training
- Data loader verification and sanity checks
- Performance tracking across training rounds
- Grad-CAM heatmaps for model interpretability

## Repository Structure

- `README.md` — project overview and setup
- `dataload_check.ipynb` — checks the data loader output, tensor shapes, and sample labels
- `dataloader.ipynb` — data loading pipeline and dataset handling
- `model.ipynb` — model architecture, federated training loop, evaluation, and Grad-CAM utilities
- `path.ipynb` — dataset folder tree inspection helper

## Dataset

This project expects a skin lesion dataset stored in a local folder such as:

```text
Dataset/
