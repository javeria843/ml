import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier  # ✅ Correct spelling
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # ✅ Correct spelling

# ✅ Correct file name (Make sure to include ".csv" if it’s the file)
df = pd.read_csv("Horror_Movies_IMDb.csv")  # file name must be correct

# ✅ Info
print("Dataset Info:")
print(df.info())

# ✅ Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# ✅ Class distribution
print("\nClass Distribution:")
print(df['Genre'].value_counts())

# ✅ Visualization
sns.countplot(x='Genre', data=df)
plt.title("Genre Distribution")
plt.xlabel("Genre")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

# ✅ Drop missing rows
df = df.dropna()

# ✅ Label Encoding
le = LabelEncoder()
df['Genre_Label'] = le.fit_transform(df['Genre'])

# ✅ Features & Target
X = df[['Action', 'Humor', 'Suspense']]  # Make sure these columns exist in your dataset
y = df['Genre_Label']

# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ KNN Model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# ✅ Prediction
y_pred = knn.predict(X_test)

# ✅ Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ✅ Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
