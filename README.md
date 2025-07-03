# Day-7-internship-task-7
 This repository contains the implementation of Support Vector Machines (SVM) using the Breast Cancer dataset. It demonstrates linear and RBF kernel classification, decision boundary visualization with PCA, hyperparameter tuning using GridSearchCV, and model evaluation with cross-validation.


# 🧠 Task 7: Support Vector Machines (SVM) - Breast Cancer Classification

This project demonstrates how to use **Support Vector Machines (SVM)** with both linear and RBF kernels on the **Breast Cancer** dataset.

---

## 📌 Objective

- Use SVMs for binary classification.
- Explore linear and non-linear kernels.
- Visualize decision boundaries.
- Tune hyperparameters (`C`, `gamma`) using GridSearch.
- Evaluate model with cross-validation.

---

## 📁 Dataset

- **Source**: `breast-cancer.csv`
- **Target**: `diagnosis` column (0 = Benign, 1 = Malignant)

---

## 🚀 Technologies Used

- Python
- NumPy, Pandas
- Scikit-learn (SVM, PCA, GridSearchCV)
- Matplotlib, Seaborn

---

## 🧪 Steps Performed

1. Load and preprocess the dataset
2. Train-test split and standardization
3. Apply SVM with linear kernel
4. Apply SVM with RBF kernel
5. PCA-based dimensionality reduction for 2D visualization
6. Hyperparameter tuning via GridSearchCV
7. 10-fold cross-validation for accuracy

---

## ✅ Results

- **Linear SVM Accuracy**: ~96%
- **RBF SVM Accuracy**: ~97%
- **Best Hyperparameters**: `C=10`, `gamma=0.01`
- **Cross-Validation Accuracy**: ~98%

---

## 📚 Interview Questions Covered

- What is a support vector?
- What does the C parameter do?
- What are kernels in SVM?
- Difference between linear and RBF kernel?
- Advantages of SVM?
- Can SVMs be used for regression?
- What happens when data is not linearly separable?
- How is overfitting handled in SVM?

---

## 📎 Files

- `svm_breast_cancer.ipynb`: Main notebook
- `breast-cancer.csv`: Dataset
- `README.md
