import matplotlib.pyplot as plt
import sys
from sklearn.cluster import KMeans

x = [2, 2, 8, 5, 7, 6, 1, 4]
y = [10, 5, 4, 8, 5, 4, 2, 9]

data = list(zip(x, y))
kmeans_model = KMeans(n_clusters=3)
kmeans_model.fit(data)
labels = kmeans_model.labels_
centroids = kmeans_model.cluster_centers_

plt.scatter(x, y, c=labels, cmap='viridis', marker='o', alpha=0.6)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, label='Centroids')
plt.title("K-Means Clustering (sklearn)")
plt.legend()
plt.show()