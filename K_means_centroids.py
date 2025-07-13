import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import pandas as pd 
import scaler as std

# Load dataset
df = pd.read_csv("customer_data.csv")

# Apply KMeans with K=3 based on Elbow result
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Get cluster centroids
centroids = kmeans.cluster_centers_

# Plot scatter with centroids
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Age', y='Spending_Score', hue='Cluster', data=df, palette='Set2', s=100)
plt.scatter(
    scaler.inverse_transform(centroids)[:, 0],
    scaler.inverse_transform(centroids)[:, 1],
    s=300, c='black', marker='X', label='Centroids'
)
plt.title('Customer Segmentation (K-Means Clusters)')
plt.xlabel('Age')
plt.ylabel('Spending Score')
plt.legend()
plt.grid(True)
plt.show()
