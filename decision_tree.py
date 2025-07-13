# 📦 Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt

# 📥 Load the Balance Scale dataset from UCI repository
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data"
    data = pd.read_csv(url, header=None)
    print("Data Loaded. Shape:", data.shape)
    return data

# ✂️ Split the dataset into features and target
def split_data(data):
    X = data.iloc[:, 1:].values  # features: columns 1 to 4
    y = data.iloc[:, 0].values   # target: column 0
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
    print("Data split into training and test sets.")
    print("Training set size:", X_train.shape[0])
    print("Test set size:", X_test.shape[0])
    return X_train, X_test, y_train, y_test

# 🧠 Train the decision tree model
def train_model(X_train, y_train, criterion="gini"):
    model = DecisionTreeClassifier(criterion=criterion, max_depth=3, min_samples_leaf=5, random_state=100)
    model.fit(X_train, y_train)
    return model

# 🔮 Make predictions
def make_predictions(model, X_test):
    return model.predict(X_test)

# 📊 Evaluate the model
def evaluate(y_test, y_pred):
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nAccuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
    print("\nPrecision:", precision_score(y_test, y_pred, average='macro'))
    print("Recall:", recall_score(y_test, y_pred, average='macro'))
    print("F1 Score:", f1_score(y_test, y_pred, average='macro'))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 🌳 Plot the decision tree
def visualize_tree(model, feature_names, class_names, title):
    plt.figure(figsize=(14, 8))
    plot_tree(model, filled=True, feature_names=feature_names, class_names=class_names, rounded=True)
    plt.title(title)
    plt.show()

# 🚀 Run everything
if __name__ == "__main__":
    data = load_data()
    X_train, X_test, y_train, y_test = split_data(data)

    # Train and evaluate with Gini Index
    print("\n--- Training with Gini Index ---")
    model_gini = train_model(X_train, y_train, criterion="gini")
    pred_gini = make_predictions(model_gini, X_test)
    evaluate(y_test, pred_gini)
    visualize_tree(model_gini, ['Left-Weight', 'Left-Distance', 'Right-Weight', 'Right-Distance'], ['L', 'B', 'R'], "Decision Tree (Gini)")

    # Train and evaluate with Entropy
    print("\n--- Training with Entropy ---")
    model_entropy = train_model(X_train, y_train, criterion="entropy")
    pred_entropy = make_predictions(model_entropy, X_test)
    evaluate(y_test, pred_entropy)
    visualize_tree(model_entropy, ['Left-Weight', 'Left-Distance', 'Right-Weight', 'Right-Distance'], ['L', 'B', 'R'], "Decision Tree (Entropy)")
