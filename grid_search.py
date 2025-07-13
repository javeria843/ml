from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import pandas as pd
import numpy as np

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Create model
model = SVC()

# Define parameter grid
parameters = {
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'C': [1, 5, 10, 20]
}

# Apply GridSearchCV
classifier = GridSearchCV(model, parameters, cv=5)
classifier.fit(X, y)

# Best parameters and best score
best_params = classifier.best_params_
best_score = classifier.best_score_

# Create results DataFrame
results_df = pd.DataFrame(classifier.cv_results_)

# Filter specific columns
grid_search_result = results_df[["param_C", "param_kernel", "mean_test_score"]]

best_params, round(best_score * 100, 2), grid_search_result.sort_values(by='mean_test_score', ascending=False).head(5)
