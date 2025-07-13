## Week 4: Model Evaluation & Hyperparameter Tuning
##Complete Solution for Cross-Validation, Confusion Matrix, and GridSearchCV using Logistic Regression on Breast Cancer Dataset.
# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
# Load breast cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target
print("------ TASK 1: K-FOLD CROSS VALIDATION ------")
model = LogisticRegression(max_iter=10000)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')

print("Accuracy scores for each fold:", scores)
print("Mean accuracy across folds:", np.mean(scores))