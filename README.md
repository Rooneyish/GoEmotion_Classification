# GoEmotions: Fine-Grained Multi-Label Emotion Classification

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-yellow)

This repository contains an academic research project focused on **Multi-Label Fine-Grained Emotion Classification**. Using Google Research’s **GoEmotions dataset**, the study compares three deep learning architectures to capture the complex, overlapping nature of human emotions in digital text.

---

## 📖 Project Overview
Traditional sentiment analysis often oversimplifies human expression into binary (positive/negative) or limited categories. This project addresses the **Fine-Grained** challenge by classifying English Reddit comments into **27 distinct emotion categories** (e.g., admiration, grief, curiosity, remorse).

### Objectives
* Implementation of a **Multi-Label** classification pipeline where one text input can trigger multiple emotion labels simultaneously.
* Comparative analysis of sequential models (**Bi-LSTM**, **Seq2Seq**) against attention-based architectures (**Encoder-only Transformer**).
* Evaluation of model performance in the context of significant **data imbalance** present in the GoEmotions dataset.

---

## 🏗️ Technical Architecture

The project follows a supervised deep learning pipeline:
1. **Preprocessing:** Text cleaning (URL/Special character removal) and normalization.
2. **Embedding:** Utilization of the pre-trained `bert-base-uncased` tokenizer for semantic representation.
3. **Hardware:** Optimization for **CUDA GPU** execution using the PyTorch framework.



### Models Implemented
* **Bi-Directional LSTM:** Captures forward and backward contextual dependencies to understand word order.
* **Seq2Seq (Encoder-Decoder):** Summarizes text into a context vector to predict multiple emotion labels.
* **Encoder-only Transformer:** Employs **Self-Attention** and **GELU activation** to capture global context across long sequences effectively.

---

## 📊 Performance Comparison

All models were trained for **30 epochs** using the **Adam Optimizer** and **BCELogitLoss**.

| Model Architecture | Macro F1-Score | Achievement |
| :--- | :---: | :--- |
| **Encoder-only Transformer** | **0.5098** | **Exceeded Google Baseline (0.46)** |
| Seq2Seq | 0.3994 | Competitive performance |
| Bi-LSTM | 0.3773 | Stable but limited context capture |

### Key Findings
* The **Transformer model** significantly outperformed recurrent models, demonstrating the efficiency of self-attention in detecting emotional nuances.
* Accuracy remains high for majority classes (e.g., *Admiration*), while minority classes (e.g., *Grief*) highlight the need for further data augmentation or specialized sampling techniques.

---
