# Fraud Detection Using Artificial Neural Networks (ANN)

This project implements a fraud detection system using an Artificial Neural Network (ANN) to identify fraudulent transactions based on historical data. The goal is to build a machine learning model that can accurately classify transactions as legitimate or fraudulent.

---

## Project Overview

Fraud detection is a critical problem in financial systems due to the high cost of fraudulent activities. This project applies a supervised learning approach using an ANN to learn complex patterns in transaction data and improve detection accuracy.

The model is trained on preprocessed transaction data and evaluated using standard classification metrics.

---

## Key Features

- Data preprocessing and normalization
- Binary classification (Fraud / Non-Fraud)
- Artificial Neural Network implementation
- Model training and evaluation
- Visualization of results using Matplotlib
- Clear and modular Python code

---

## Tech Stack

- **Programming Language:** Python  
- **Machine Learning:** Artificial Neural Networks (ANN)  
- **Libraries:**
  - NumPy
  - Pandas
  - Scikit-learn
  - TensorFlow / Keras
  - Matplotlib

---

## Dataset

- The dataset consists of transaction-level data with multiple numerical features.
- The target variable indicates whether a transaction is fraudulent.
- Data preprocessing includes:
  - Handling missing values
  - Feature scaling
  - Train-test split

*(Dataset source can be added here if public.)*

---

## Model Architecture

- Input Layer: Based on the number of features
- Hidden Layers:
  - Fully connected dense layers
  - ReLU activation
- Output Layer:
  - Sigmoid activation for binary classification

The model is trained using:
- Binary Cross-Entropy loss
- Adam optimizer

---

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

These metrics help assess the model’s effectiveness in detecting fraudulent transactions while minimizing false positives.

---

## Visualizations

The project uses **Matplotlib** to visualize:
- Training and validation loss
- Training and validation accuracy
- Confusion matrix

These plots help in understanding model performance and convergence.

---


   
