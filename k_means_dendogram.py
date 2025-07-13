import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import pandas as pd 

# Load dataset
df = pd.read_csv("customer_data.csv")
# Apply Hierarchical Clustering
linked = linkage(X_scaled, method='ward')

# Plot Dendrogram
plt.figure(figsize=(10, 5))
dendrogram(linked,
           orientation='top',
           labels=df['CustomerID'].values,
           distance_sort='descending',
           show_leaf_counts=True)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Customer ID')
plt.ylabel('Distance')
plt.grid(True)
plt.show()
