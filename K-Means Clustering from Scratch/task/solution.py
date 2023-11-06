import numpy as np
import sklearn.metrics
from sklearn.datasets import load_wine
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.metrics import silhouette_score

# scroll down to the bottom to implement your solution
class CustomKMeans:
    def __init__(self, k):
        self.k = k
        self.centres = None

    def fit(self, X, eps=1e-6):
        # Take the first k objects of X as initial cluster centers
        self.centres = X[:self.k]

        while True:
            # Assign data point to the nearest cluster center
            cluster_labels = find_nearest_center(X, self.centres)

            # Calculate new cluster centers
            new_centers = calculate_new_centers(cluster_labels, X, self.k)

            # Check if the algorithm has converged
            if np.all(np.linalg.norm(new_centers - self.centres, axis=1) < eps):
                break

            self.centres = new_centers

    def predict(self, X):
        # Assign data points to the nearest cluster center using the final centers
        cluster_labels = find_nearest_center(X, self.centres)
        return cluster_labels


def find_nearest_center(x, centers):
    distances = pairwise_distances_argmin_min(x, centers)
    cluster_labels = distances[0]
    return cluster_labels


def calculate_new_centers(cluster_labels, X_full, num_clusters):
    new_centers = []
    for i in range(num_clusters):
        cluster_points = X_full[cluster_labels == i]
        if len(cluster_points) > 0:
            new_center = np.mean(cluster_points, axis=0)
            new_centers.append(new_center)

    return np.array(new_centers)


def calculate_inertia(X, cluster_labels, centers):
    inertia = 0
    for i in range(len(centers)):
        cluster_points = X[cluster_labels == i]
        cluster_center = centers[i]
        inertia += np.sum(np.linalg.norm(cluster_points - cluster_center, axis=1) ** 2)
    return inertia


def plot_comparison(data: np.ndarray, predicted_clusters: np.ndarray, true_clusters: np.ndarray = None,
                    centers: np.ndarray = None, show: bool = True):
    # Use this function to visualize the results on Stage 6.

    if true_clusters is not None:
        plt.figure(figsize=(20, 10))

        plt.subplot(1, 2, 1)
        sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=predicted_clusters, palette='deep')
        if centers is not None:
            sns.scatterplot(x=centers[:, 0], y=centers[:, 1], marker='X', color='k', s=200)
        plt.title('Predicted clusters')
        plt.xlabel('alcohol')
        plt.ylabel('malic_acid')
        plt.grid()

        plt.subplot(1, 2, 2)
        sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=true_clusters, palette='deep')
        if centers is not None:
            sns.scatterplot(x=centers[:, 0], y=centers[:, 1], marker='X', color='k', s=200)
        plt.title('Ground truth')
        plt.xlabel('alcohol')
        plt.ylabel('malic_acid')
        plt.grid()
    else:
        plt.figure(figsize=(10, 10))
        sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=predicted_clusters, palette='deep')
        if centers is not None:
            sns.scatterplot(x=centers[:, 0], y=centers[:, 1], marker='X', color='k', s=200)
        plt.title('Predicted clusters')
        plt.xlabel('alcohol')
        plt.ylabel('malic_acid')
        plt.grid()

    plt.savefig('Visualization.png', bbox_inches='tight')
    if show:
        plt.show()


if __name__ == '__main__':
    # Load data
    data = load_wine(as_frame=True, return_X_y=True)
    X_full, y_full = data

    # Permutate it to make things more interesting
    rnd = np.random.RandomState(42)
    permutations = rnd.permutation(len(X_full))
    X_full = X_full.iloc[permutations]
    y_full = y_full.iloc[permutations]

    # From dataframe to ndarray
    X_full = X_full.values
    y_full = y_full.values

    # Scale data
    scaler = StandardScaler()
    X_full = scaler.fit_transform(X_full)

    # Take the first three data points as initial cluster centers
    cluster_centers = X_full[:3]

    # num_clusters = 3
    # new_centers = calculate_new_centers(cluster_labels, X_full, num_clusters)
    #
    # new_centers_list = new_centers.tolist()
    # flatten_list = [item for sublist in new_centers_list for item in sublist]

    # Print the resulting coordinates of the centers for the entire dataset
    # print(flatten_list)

    cluster_labels = find_nearest_center(X_full, cluster_centers)
    cluster_labels = cluster_labels.tolist()

    # Range of k values to test
    k_values = range(2, 11)

    inertias = []
    silhoutte_score = []

    for k in k_values:
        kmeans = CustomKMeans(k=k)
        kmeans.fit(X_full, eps=1e-6)
        cluster_labels = kmeans.predict(X_full)
        # inertia = calculate_inertia(X_full, cluster_labels, kmeans.centres)
        # inertias.append(inertia)
        silhoutte_score.append(silhouette_score(X_full, cluster_labels))
    # print(inertias)


# Determine the optimal k based on the silhouette scores
optimal_k = k_values[np.argmax(silhoutte_score)]

# Create an instance of your custom K-Means model with the optimal k
kmeans = CustomKMeans(k=optimal_k)

# Fit the model on the data X_full
kmeans.fit(X_full, eps=1e-6)

# Make predictions for all objects in X_full
predictions = kmeans.predict(X_full)

# Print the results for the first 20 objects
print(list(predictions[:20]))
