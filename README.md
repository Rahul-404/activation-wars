# ⚔️ Activation Wars: Comparing Activation Functions in Neural Networks

This project explores the impact of different activation functions on neural network performance.  
We train and compare models using sigmoid, tanh, ReLU, Leaky ReLU, Parametric ReLU, ELU, Swish, and GELU on several tasks.

## 🔍 Overview

🧠 Goal:

Compare activation functions across 3 datasets (MNIST, Fashion MNIST, CIFAR-10) in terms of:

- Accuracy
- Loss
- Convergence speed
- Gradient flow
- Weight & bias evolution (via TensorBoard)

## 🧠 Activation Functions Compared

| Activation | Pros | Cons |
|------------|------|------|
| Sigmoid    | Smooth, probabilistic | Vanishing gradients, slow |
| Tanh       | Zero-centered         | Still vanishes for large values |
| ReLU       | Fast, sparse          | Can "die" (neurons stop updating) |
| Leaky ReLU | Fixes ReLU dying      | Slight added complexity |
| Swish      | Smooth + non-linear   | Slower to compute |
| GELU       | State-of-the-art in NLP | Non-trivial implementation |

## 🗂 Structure

- `notebooks/` – Main analysis notebook
- `utils/` – Reusable code: training loops, activations
- `results/` – Plots and metrics from experiments

## 📈 Sample Plot

![activation-comparison](results/plots/loss_comparison.png)

## 🛠 Setup

```bash
git clone https://github.com/Rahul-404/activation-wars.git
cd activation-wars
pip install -r requirements.txt
```

🚀 Launch TensorBoard to Compare

```bash
tensorboard --logdir=logs
```

In your browser:

```bash
http://localhost:6006
```

