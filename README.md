# Step Size Strategies in Stochastic Gradient Descent for Neural Network Training


## 📌 Overview

This repository contains the code and experiments for my Master's thesis in Data Science, investigating the impact of different step size (learning rate) strategies on Stochastic Gradient Descent (SGD) for neural network training. The research compares how constant step sizes, decreasing step sizes, Armijo line-search, and momentum-based approaches affect training dynamics, convergence speed, and final classification performance.

## 🎯 Research Question

How do different step size strategies influence the convergence behavior and practical performance of SGD when training neural networks on image classification tasks?

## 🧠 Models Implemented

| Model | Architecture | Key Features |
|-------|-------------|--------------|
| **MLP** | 784 → 1000 → 10 | Single hidden layer baseline |
| **DeepMLP** | 784 → 2048 → 1024 → 512 → 256 → 10 | Deeper fully-connected network |
| **DeepMLPBN** | Same as DeepMLP + BatchNorm + Dropout | Regularization & training stability |
| **CNN** | 3× (Conv2d + BatchNorm + ReLU + MaxPool) → Classifier | Spatial feature extraction (32→64→128 filters) |

## 📊 Step Size Strategies

- **Constant step size** – Fixed learning rate throughout training
- **Decay step size** – Learning rate decreases according to predefined schedule
- **Armijo line search** – Adaptive step size satisfying Armijo condition
- **SGD with Momentum** – Accelerates convergence using accumulated gradients

## 🔬 Experiments

Each model was trained with **all four step size strategies** for:
- 30 epochs (short-term convergence analysis)
- 60 epochs (final performance comparison)

Total experiments: **4 models × 4 strategies × 2 durations = 32 training runs**


## 🚀 Getting Started


## Installation

1. Clone the repository:
```bash
git clone git@github.com:bdmonir/SGD-step-size-strategies.git
cd SGD-step-size-strategies
```
2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
3. Running Experiments
```bash
python main.py
