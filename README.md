# Deploymen of the project "AI-enabled harm assessment system: Utilizing Mamba Architecture for Identifying Unconscious Individuals and Assessing Potential Harm through AI Models"

**PAPER PUBLISHED FOR ACTION RECOGNITION**

> **SkeletonMamba: a Lightweight Mamba-based Architecture for Action Recognition**  
> Sanzhar Abdrakhim and Luca Rossi  
> The Hong Kong Polytechnic University  

## Abstract

In the past few years, graph neural networks (GNNs) have become the preferred way to tackle the computer vision task of action recognition using skeleton data. These are usually combined with temporal operations aimed at capturing the dynamics induced by the sequence of frames. Recent works have proposed to use Mamba, a recently introduced sequence modeling architecture that offers a fast and computationally efficient alternative to Transformers, to capture these temporal dynamics, while employing GNNs to capture spatial information. Unfortunately, these approaches fail to fully untap the computational efficiency of Mamba and only achieve substandard accuracy. In this paper, we propose SkeletonMamba, a novel hybrid GNN-Mamba model combining a lightweight GNN module with a novel Temporal Mamba block. The synergy between these two components allows our model to overcome to shortcomings of existing Mamba-based action recognition architectures. We perform an extensive set of experiment demonstrating that SkeletonMamba achieves a competitive performance against SOTA approaches on the widely used NTU RGB+D 60 \& 120 datasets, while minimizing the number of floating point operations and trainable parameters.


## Installation

```bash
# 1. Create and activate a clean environment
conda create -n skeletonmamba python=3.10 -y
conda activate skeletonmamba

# 2. Install all required packages in one command
pip install -r requirements.txt

# 3. Install Mamba-SSM packages separately with compatible versions (if using edge platform, e.g. jetson, a wheel building will be required)
#    (these often need --no-build-isolation due to build requirements)
pip install causal-conv1d==1.5.0.post8 --no-build-isolation
pip install mamba-ssm==2.2.4 --no-build-isolation
pip install mmcv==2.1.0 --no-build-isolation
```

## Usage
To use the harm assesment system, you need to download necessary packages and ensure their compatability (some versions are conflicting).
To start the program, run:
```bash
python harm_assessment.py
```
## Training, Evaluation, and Testing (FLOPS & Parameters
Instructions can be founded in main branch of this repo.
