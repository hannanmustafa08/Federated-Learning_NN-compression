# Neural Network Compression & Optimization

This repository contains implementations of advanced model compression techniques designed to reduce the computational footprint and memory requirements of Deep Neural Networks (DNNs) without significant degradation in accuracy. 

Developed as part of the CS437 Deep Learning coursework at LUMS, this project focuses on optimizing ResNet architectures trained on the CIFAR-10 dataset using two primary techniques: **Magnitude-Based Pruning** and **Knowledge Distillation**.

## 🌐 Federated Learning Framework

Built as the first component of this project — a complete federated learning infrastructure from scratch, without using any FL library (no Flower, no TensorFlow Federated).

- **FedAvg:** Implemented the canonical Federated Averaging algorithm across 10 simulated clients with non-IID Dirichlet-distributed data (α=0.5), conducting a full client heterogeneity audit.
- **FedProx:** Extended FedAvg with a proximal term (μ) to stabilise training under high data heterogeneity.
- **Real-World Application:** Applied the framework to the Distracted Driver Detection dataset (SFD3) with fairness-aware aggregation design.

The compression work (pruning + distillation) was then applied to models trained within this federated setting.

## 🛠️ Tech Stack
* **Framework:** PyTorch, torchvision
* **Language:** Python
* **Data Processing & Visualization:** NumPy, Matplotlib
* **Dataset:** CIFAR-10

## 🚀 Key Implementations

### 1. Magnitude-Based Pruning
* **Unstructured Pruning:** Implemented L1/L2 magnitude-based weight pruning from scratch to eliminate low-salience parameters across the network.
* **Structured Pruning:** Applied channel-wise structured pruning to physically reduce the dimensions of convolutional layers, analyzing the direct implications on hardware acceleration and memory bandwidth.
* **Sparsity vs. Accuracy Trade-off:** Conducted iterative prune-and-finetune cycles to map the degradation curve and identify the optimal sparsity threshold.

### 2. Knowledge Distillation (Teacher-Student Architecture)
* **Logit Matching (Hinton et al.):** Trained a lightweight "Student" ResNet model to mimic the softened output probabilities (logits) of a massive, pre-trained "Teacher" network using Kullback-Leibler (KL) divergence loss and a tunable temperature parameter.
* **FitNets (Hint-Based Distillation):** Implemented intermediate feature-map distillation, forcing the student's hidden layers to learn the representational space of the teacher's hidden layers using guided regression and projection layers.

## 📂 Repository Structure
* `1_Magnitude_Pruning_ResNet.ipynb`: Contains the training loop, pruning algorithms, and sparsity analysis for both structured and unstructured approaches.
* `2_Knowledge_Distillation_FitNets.ipynb`: Implements the Teacher-Student training pipelines, including temperature scaling and intermediate representation matching.

## 💻 How to Run
Ensure you have a CUDA-enabled GPU for optimal training times.
1. Clone the repository.
2. Install dependencies: `pip install torch torchvision numpy matplotlib`
3. Launch Jupyter and run the notebooks sequentially. The CIFAR-10 dataset will be downloaded automatically via `torchvision.datasets` upon first execution.
