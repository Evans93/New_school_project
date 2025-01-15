import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler

# Load dataset
# Replace 'house_data.csv' with your actual dataset file
file_path = 'house_prices.csv'  # Update with your file path
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print("Dataset Preview:")
print(data.head())

# Select features for clustering
# Replace 'Size', 'Bedrooms', and 'Price' with actual column names from your dataset
selected_columns = ['size', 'bedrooms', 'price']  # Adjust columns based on your dataset
data_points = data[selected_columns].values

# Standardize the data to improve clustering performance
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_points)

# Perform K-Means clustering
num_clusters = 3  # Adjust the number of clusters as needed
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(scaled_data)

# Calculate evaluation metrics
silhouette_avg = silhouette_score(scaled_data, cluster_labels)
davies_bouldin_avg = davies_bouldin_score(scaled_data, cluster_labels)

# Print metrics
print(f"Silhouette Score: {silhouette_avg:.2f}")
print(f"Davies-Bouldin Score: {davies_bouldin_avg:.2f}")

# Add cluster labels to the original dataset
data['Cluster'] = cluster_labels

# Display dataset with cluster labels
print("\nDataset with Clusters:")
print(data.head())

# Extract cluster centers (scaled back to original units)
cluster_centers_scaled = kmeans.cluster_centers_
cluster_centers = scaler.inverse_transform(cluster_centers_scaled)

# Plot clusters (using the first two features for simplicity)
plt.figure(figsize=(8, 6))
for label in np.unique(cluster_labels):
    cluster_points = scaled_data[cluster_labels == label]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {label}', s=100)

# Plot cluster centers
plt.scatter(
    cluster_centers_scaled[:, 0], cluster_centers_scaled[:, 1],
    c='red', label='Cluster Centers', s=200, marker='X'
)

plt.title("K-Means Clustering for House Data")
plt.xlabel(selected_columns[0])
plt.ylabel(selected_columns[1])
plt.legend()
plt.grid()
plt.show()