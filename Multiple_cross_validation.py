from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Define models
models = [
    LogisticRegression(max_iter=1000),
    SVC(kernel='linear'),
    KNeighborsClassifier(),
    RandomForestClassifier()
]

model_names = ['Logistic Regression', 'SVM', 'KNN', 'Random Forest']

# Loop through models
for name, model in zip(model_names, models):
    scores = cross_val_score(model, X, y, cv=5)
    mean_accuracy = round(scores.mean() * 100, 2)
    print(f"\nModel: {name}")
    print("Scores:", scores)
    print("Average Accuracy:", mean_accuracy, "%")
