import matplotlib.pyplot as plt
import seaborn as sns

# Sample top 5 results from GridSearchCV for demonstration (simulated based on expected format)
import pandas as pd

grid_search_result = pd.DataFrame({
    'param_C': [10, 5, 20, 1, 10],
    'param_kernel': ['linear', 'linear', 'linear', 'linear', 'rbf'],
    'mean_test_score': [0.9702, 0.9684, 0.9666, 0.9649, 0.6274]
})

# Bar Plot
plt.figure(figsize=(10, 5))
sns.barplot(
    data=grid_search_result,
    x='param_C',
    y='mean_test_score',
    hue='param_kernel'
)
plt.title('Top 5 Kernel and C Combinations by Accuracy')
plt.ylabel('Mean CV Accuracy')
plt.xlabel('C Value')
plt.ylim(0.6, 1.0)
plt.legend(title='Kernel')
plt.tight_layout()
plt.show()

# Heatmap
# Pivoting the data to format suitable for heatmap
heatmap_data = grid_search_result.pivot("param_kernel", "param_C", "mean_test_score")

plt.figure(figsize=(8, 6))
sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", fmt=".4f")
plt.title('Mean CV Accuracy Heatmap')
plt.xlabel('C Value')
plt.ylabel('Kernel')
plt.tight_layout()
plt.show()
