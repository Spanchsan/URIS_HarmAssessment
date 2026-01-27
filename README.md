# SkeletonMamba: a Lightweight Mamba-based Architecture for Action Recognition

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)

**Official PyTorch implementation** of the paper:

> **SkeletonMamba: a Lightweight Mamba-based Architecture for Action Recognition**  
> Sanzhar Abdrakhim and Luca Rossi  
> The Hong Kong Polytechnic University  

## Abstract

In the past few years, graph neural networks (GNNs) have become the preferred way to tackle the computer vision task of action recognition using skeleton data. These are usually combined with temporal operations aimed at capturing the dynamics induced by the sequence of frames. Recent works have proposed to use Mamba, a recently introduced sequence modeling architecture that offers a fast and computationally efficient alternative to Transformers, to capture these temporal dynamics, while employing GNNs to capture spatial information. Unfortunately, these approaches fail to fully untap the computational efficiency of Mamba and only achieve substandard accuracy. In this paper, we propose SkeletonMamba, a novel hybrid GNN-Mamba model combining a lightweight GNN module with a novel Temporal Mamba block. The synergy between these two components allows our model to overcome to shortcomings of existing Mamba-based action recognition architectures. We perform an extensive set of experiment demonstrating that SkeletonMamba achieves a competitive performance against SOTA approaches on the widely used NTU RGB+D 60 \& 120 datasets, while minimizing the number of floating point operations and trainable parameters.

## Key Features

- State-of-the-art efficiency–accuracy trade-off for skeleton-based action recognition
- Pure Mamba temporal backbone (no redundant TCN blocks)
- Lightweight GNN with static + dynamic topology modeling
- Fully implemented in PyTorch with support for NTU RGB+D datasets
- Easy to train and evaluate (notebook-based workflow)

## Results

Preliminary results on NTU RGB+D (as reported in the paper):

| Dataset          | Protocol     | Top-1 Accuracy | Params (M) | FLOPs (G) |
|------------------|--------------|----------------|------------|-----------|
| NTU RGB+D 60     | Cross-Subject| XX.X%          | ~0.XX      | ~X.X      |
| NTU RGB+D 60     | Cross-View   | XX.X%          | ~0.XX      | ~X.X      |
| NTU RGB+D 120    | Cross-Subject| XX.X%          | ~0.XX      | ~X.X      |
| NTU RGB+D 120    | Cross-Setup  | XX.X%          | ~0.XX      | ~X.X      |

*(Numbers will be updated once final paper results are available)*

## Installation

```bash
# 1. Create and activate a clean environment
conda create -n skeletonmamba python=3.10 -y
conda activate skeletonmamba

# 2. Install all required packages in one command
pip install -r requirements.txt

# 3. Install Mamba-SSM packages separately with compatible versions
#    (these often need --no-build-isolation due to build requirements)
pip install causal-conv1d==1.5.0.post8 --no-build-isolation
pip install mamba-ssm==2.2.4 --no-build-isolation

# Optional: for FLOPs calculation
pip install ptflops
```

## Data Preparation
1. Download the NTU RGB+D datasets (60 & 120) from the official sources.
2. Preprocess the data into .pkl format (joint coordinates, bone, velocity if used).
3. Place processed files in a directory (e.g., data/ntu/).

Our **feeder** (feeders/feeder_ntu.py) expects the standard NTU skeleton data format.

## Usage
The main training and evaluation code is provided in notebook.ipynb (Jupyter Notebook).

## Training
Open notebook.ipynb and run cells sequentially. Key parameters to adjust:
```python
epochs = 140
batch_size = 32
out_channels = 80          # model width
dataset_t = 'ntu60'        # or 'ntu120'
optimz = 'SGD'             # or 'AdamW'
seed = 2
```
## Evaluation / Testing
The same notebook contains evaluation code that reports top-1/top-5 accuracy, confusion matrix, and per-class performance.
## FLOPs & Parameters
```python
from ptflops import get_model_complexity_info
Flops, params = get_model_complexity_info(model, (3, 64, 25, 2), as_strings=True)
print(f"FLOPs: {Flops}, Params: {params}")
```
