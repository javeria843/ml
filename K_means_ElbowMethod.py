import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import pandas as pd 

# Load dataset
df = pd.read_csv("customer_data.csv")

# We'll use Age and Spending_Score for clustering
X = df[['Age', 'Spending_Score']]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Elbow Method ---
wcss = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot Elbow
plt.figure(figsize=(8, 4))
plt.plot(K_range, wcss, 'bo-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.title('Elbow Method for Optimal K')
plt.grid(True)
plt.show()
