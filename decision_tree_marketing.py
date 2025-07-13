from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import pandas as pd
import matplotlib.pyplot as plt  # ✅ Corrected import

# 📥 Load the dataset
data = pd.read_csv("marketing_campaign.csv")

# 🎯 Features and target
X = data[["Age", "Income"]]
y = data["Subscribed"]

# 🧪 Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)  # ✅ Fixed test_size

# 🌳 Train the decision tree
model = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
model.fit(X_train, y_train)

# 🔮 Make predictions
y_pred = model.predict(X_test)

# 📊 Evaluate the model
evaluation = {
    "Accuracy": round(accuracy_score(y_test, y_pred), 2),
    "Precision": round(precision_score(y_test, y_pred), 2),
    "Recall": round(recall_score(y_test, y_pred), 2),
    "F1 Score": round(f1_score(y_test, y_pred), 2),
    "Confusion Matrix": confusion_matrix(y_test, y_pred).tolist(),
    "Classification Report": classification_report(y_test, y_pred, output_dict=True)
}

# 🌲 Visualize the decision tree
plt.figure(figsize=(10, 6))
plot_tree(model, feature_names=["Age", "Income"], class_names=["Not Subscribed", "Subscribed"], filled=True, rounded=True)
plt.title("Decision Tree for Marketing Campaign")
plt.tight_layout()
plt.show()

# 📋 Print evaluation
print(evaluation)
