from sklearn.model_selection import KFold, cross_val_score
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import numpy as np
data = load_iris()
X, y = data.data, data.target
k = 5  
kf = KFold(n_splits=k, shuffle=True, random_state=42)
model = RandomForestClassifier(random_state=42)
scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
print(f"Accuracy for each fold: {scores}")
average_accuracy = np.mean(scores)
print(f"Average Accuracy: {average_accuracy:.2f}")
