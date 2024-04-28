"""
Work with Jarvis-Patrick clustering.
Do not use global variables!
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import pickle
import scipy
from scipy.spatial.distance import cdist

######################################################################
#####     CHECK THE PARAMETERS     ########
######################################################################

def confusion_matrix(true_labels, predicted_labels): ## implemented confusion matrix
    """
    Compute the confusion matrix for a two-class problem.

    Parameters:
    - true_labels: The true labels of the data points.
    - predicted_labels: The predicted labels of the data points.

    Returns:
    - A 2x2 numpy array representing the confusion matrix.
      [[true positive, false negative],
       [false positive, true negative]]
    """
    # Initialize the confusion matrix to zeros
    confusion = np.zeros((2, 2), dtype=int)

    # True positives (TP)
    confusion[0, 0] = np.sum((true_labels == 1) & (predicted_labels == 1))

    # True negatives (TN)
    confusion[1, 1] = np.sum((true_labels == 0) & (predicted_labels == 0))

    # False positives (FP)
    confusion[1, 0] = np.sum((true_labels == 0) & (predicted_labels == 1))

    # False negatives (FN)
    confusion[0, 1] = np.sum((true_labels == 1) & (predicted_labels == 0))

    return confusion

def compute_SSE(data, labels):
    """
    Calculate the sum of squared errors (SSE) for a clustering.

    Parameters:
    - data: numpy array of shape (n, 2) containing the data points
    - labels: numpy array of shape (n,) containing the cluster assignments

    Returns:
    - sse: the sum of squared errors
    """
    sse = 0.0
    for i in np.unique(labels):
        cluster_points = data[labels == i]
        cluster_center = np.mean(cluster_points, axis=0)
        sse += np.sum((cluster_points - cluster_center) ** 2)
    return sse

def adjusted_rand_index(labels_true, labels_pred) -> float:
    """
    Compute the adjusted Rand index.

    Parameters:
    - labels_true: The true labels of the data points.
    - labels_pred: The predicted labels of the data points.

    Returns:
    - ari: The adjusted Rand index value.

    The adjusted Rand index is a measure of the similarity between two data clusterings.
    It takes into account both the similarity of the clusters themselves and the similarity
    of the data points within each cluster. The adjusted Rand index ranges from -1 to 1,
    where a value of 1 indicates perfect agreement between the two clusterings, 0 indicates
    random agreement, and -1 indicates complete disagreement.
    """
    # Create contingency table
    contingency_table = np.histogram2d(
        labels_true,
        labels_pred,
        bins=(np.unique(labels_true).size, np.unique(labels_pred).size),
    )[0]

    # Sum over rows and columns
    sum_combinations_rows = np.sum(
        [np.sum(nj) * (np.sum(nj) - 1) / 2 for nj in contingency_table]
    )
    sum_combinations_cols = np.sum(
        [np.sum(ni) * (np.sum(ni) - 1) / 2 for ni in contingency_table.T]
    )

    # Sum of combinations for all elements
    N = np.sum(contingency_table)
    sum_combinations_total = N * (N - 1) / 2

    # Calculate ARI
    ari = (
        np.sum([np.sum(n_ij) * (np.sum(n_ij) - 1) / 2 for n_ij in contingency_table])
        - (sum_combinations_rows * sum_combinations_cols) / sum_combinations_total
    ) / (
        (sum_combinations_rows + sum_combinations_cols) / 2
        - (sum_combinations_rows * sum_combinations_cols) / sum_combinations_total
    )

    return ari

#extract random samples from the data
def extract_samples(
    data: NDArray[np.floating], labels: NDArray[np.int32], num_samples: int
) -> tuple[NDArray[np.floating], NDArray[np.int32]]:
    """
    Extract random samples from data and labels.

    Arguments:
    - data: numpy array of shape (n, 2)
    - labels: numpy array of shape (n,)
    - num_samples: number of samples to extract

    Returns:
    - data_samples: numpy array of shape (num_samples, 2)
    - label_samples: numpy array of shape (num_samples,)
    """
    indices = np.random.choice(data.shape[0], size=num_samples, replace=False)
    data_samples = data[indices]
    label_samples = labels[indices]
    return data_samples, label_samples

def jarvis_patrick(
    data: NDArray[np.floating], labels: NDArray[np.int32], params_dict: dict
) -> tuple[NDArray[np.int32] | None, float | None, float | None]:
    """
    Implementation of the Jarvis-Patrick algorithm only using the `numpy` module.

    Arguments:
    - data: a set of points of shape 50,000 x 2.
    - dict: dictionary of parameters. The following two parameters must
       be present: 'k', 'smin', There could be others.
    - params_dict['k']: the number of nearest neighbors to consider. This determines the size of the neighborhood used to assess the similarity between datapoints. Choose values in the range 3 to 8
    - params_dict['smin']:  the minimum number of shared neighbors to consider two points in the same cluster.
       Choose values in the range 4 to 10.

    Return values:
    - computed_labels: computed cluster labels
    - SSE: float, sum of squared errors
    - ARI: float, adjusted Rand index

    Notes:
    - the nearest neighbors can be bidirectional or unidirectional
    - Bidirectional: if point A is a nearest neighbor of point B, then point B is a nearest neighbor of point A).
    - Unidirectional: if point A is a nearest neighbor of point B, then point B is not necessarily a nearest neighbor of point A).
    - In this project, only consider unidirectional nearest neighboars for simplicity.
    - The metric  used to compute the the k-nearest neighberhood of all points is the Euclidean metric
    """

    k = params_dict['k']
    smin = params_dict['smin']

     # Compute the pairwise Euclidean distances
    distances = cdist(data, data, 'euclidean')

    # Find the k nearest neighbors (excluding the point itself)
    neighbors = np.argsort(distances, axis=1)[:, 1:k+1]

    # Initialize labels as -1 (unassigned)
    computed_labels = np.full(data.shape[0], -1, dtype=int)
    cluster_id = 0

    # Define clusters based on shared nearest neighbors
    for i in range(data.shape[0]):
        if computed_labels[i] == -1:  # Point not yet assigned to a cluster
            cluster_points = [i]
            for j in range(data.shape[0]):
                if i != j and computed_labels[j] == -1:
                    # Count shared neighbors
                    shared_neighbors = np.intersect1d(neighbors[i], neighbors[j]).size
                    if shared_neighbors >= smin:
                        cluster_points.append(j)
            if len(cluster_points) > 1:
                computed_labels[cluster_points] = cluster_id
                cluster_id += 1

    SSE = compute_SSE(data, computed_labels)

    ARI = adjusted_rand_index(labels, computed_labels)

    computed_labels: NDArray[np.int32] #| None = None
    SSE: float #| None = None
    ARI: float #| None = None

    return computed_labels, SSE, ARI


def jarvis_patrick_clustering(random_state = 42):
    """
    Performs Jarvis-Patrick clustering on a dataset.

    Returns:
        answers (dict): A dictionary containing the clustering results.
    """

    answers = {}

    # Return your `jarvis_patrick` function
    answers["jarvis_patrick_function"] = jarvis_patrick

    # Work with the first 10,000 data points: data[0:10000]
    # Do a parameter study of this data using Jarvis-Patrick.
    # Minimmum of 10 pairs of parameters ('sigma' and 'xi').

    # Create a dictionary for each parameter pair ('sigma' and 'xi').
    groups = {}

    data = np.load("question1_cluster_data.npy")
    labels = np.load("question1_cluster_labels.npy")
    np.random.seed(random_state)
    print(f"Data shape: {data.shape}")
    print(f"Labels shape: {labels.shape}")

    data_samples, label_samples = extract_samples(data, labels, 1000)
    print(f"Data shape: {data_samples.shape}")
    print(f"Labels shape: {label_samples.shape}")

    nb_trials = 10
    nb_samples = 1000

    k_values = range(6, 9)  # k values from 3 to 8
    smin_values = range(4, 9)  # smin values from 4 to 10

    best_k = None
    best_smin = None
    best_ari = -1

    # Find best k by testing each k value independently
    for k in k_values:
        computed_labels, SSE, ARI = jarvis_patrick(data_samples, label_samples, {'k': k, 'smin': 6})  # Fixed smin for testing k
        print(f"Testing k={k}, ARI={ARI}")
        if ARI > best_ari:
            best_ari = ARI
            best_k = k
    print(f"Best k value on the initial slice: {best_k} with ARI: {best_ari}")

    # Reset best_ari to find best smin
    best_ari = -1

    # Find best smin by testing each smin value independently
    for smin in smin_values:
        computed_labels, SSE, ARI = jarvis_patrick(data_samples, label_samples, {'k': best_k, 'smin': smin})
        print(f"Testing smin={smin}, ARI={ARI}")
        if ARI > best_ari:
            best_ari = ARI
            best_smin = smin
    print(f"Best smin value on the initial slice: {best_smin} with ARI: {best_ari}")

    slices_results = []
    for i in range(10):
        # Assuming data is already shuffled or slices are taken randomly
        data_slice = data[1000*i:1000*(i+1)]
        label_slice = labels[1000*i:1000*(i+1)]
        computed_labels, SSE, ARI = jarvis_patrick(data_slice, label_slice, {'k': best_k, 'smin': best_smin})
        groups[i] = {"k": best_k, "smin": best_smin, "ARI": ARI, "SSE": SSE}
        slices_results.append((computed_labels, SSE, ARI))

    # Process the results for the 10 slices
    # For example, you could calculate the mean and standard deviation of the ARI across the 10 slices
    ari_values = [result[2] for result in slices_results]
    mean_ari = np.mean(ari_values)
    std_ari = np.std(ari_values)

    print(f"Standard deviation of ARI across the 10 slices: {std_ari}")

    print(f"The groups are : {groups}")

    sse_values = []
    ari_values = []

    # Collect SSE and ARI values from each group in the 'groups' dictionary
    for i in groups:
        sse_values.append(groups[i]['SSE'])
        ari_values.append(groups[i]['ARI'])

    # Calculate the mean and standard deviation for SSE and ARI
    mean_sse = np.mean(sse_values)
    std_sse = np.std(sse_values)
    mean_ari = np.mean(ari_values)
    std_ari = np.std(ari_values)

    # Print the results
    print(f"Mean SSE: {mean_sse}")
    print(f"Standard Deviation of SSE: {std_sse}")
    print(f"Mean ARI: {mean_ari}")
    print(f"Standard Deviation of ARI: {std_ari}")

    
    best_ari_sigma = max(groups, key=lambda x: groups[x]['ARI'])
    best_sse_sigma = min(groups, key=lambda x: groups[x]['SSE'])

    def plot_clustering(data, labels, title):
        plt.figure(figsize=(8, 6))
        plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k', s=50)
        plt.title(title)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.grid(True)
        plt.colorbar(label='Cluster Label')
        plt.savefig("Question_1_Jarvis_Patrick_Clustering_Result_for_all_points.pdf")
        plt.show()
        
    
    plot_clustering(data_slice, computed_labels, "Question 1 - Jarvis Patrick - Clustering Result for 1000 random points")

    # Print the best sigma values for debugging
    print("Sigma with the largest ARI:", best_ari_sigma, "with ARI:", groups[best_ari_sigma]['ARI'])
    print("Sigma with the smallest SSE:", best_sse_sigma, "with SSE:", groups[best_sse_sigma]['SSE'])

    def plot_cluster_results_ARI(data, labels, title):
        plt.figure(figsize=(10, 8))
        plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', edgecolor='k', s=50)
        plt.title(title)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.grid(True)
        plt.colorbar(label='Cluster Label')
        plt.savefig("Question_1_Jarvis_Patrick_Clustering_Result_with_Largest_ARI.pdf")
        plt.show()
        

    def plot_cluster_results_SSE(data, labels, title):
        plt.figure(figsize=(10, 8))
        plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', edgecolor='k', s=50)
        plt.title(title)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.grid(True)
        plt.colorbar(label='Cluster Label')
        plt.savefig("Question_1_Jarvis_Patrick_Clustering_Result_with_smallest_SSE.pdf")
        plt.show()
        
        

    # Assuming `data` is loaded and `spectral` function is available
    data_samples, label_samples = extract_samples(data, labels, 1000)  # Adjust number of samples as needed

    # Cluster with largest ARI
    computed_labels, _, _ = jarvis_patrick(data_samples, label_samples, {'k': 8, 'smin' : 4}) ## change here
    plot_cluster_results_ARI(data_samples, computed_labels, f"Question 1 - Jarvis Patrick - Clustering Result with Largest ARI for 1000 points(Sigma={best_ari_sigma})")
    

    # Cluster with smallest SSE
    computed_labels, _, _= jarvis_patrick(data_samples, label_samples, {'k': 8, 'smin' : 4}) ## change here
    plot_cluster_results_SSE(data_samples, computed_labels, f"Question 1 - Jarvis Patrick - Clustering Result with Smallest SSE for 1000 points(Sigma={best_sse_sigma})")

    # data for data group 0: data[0:10000]. For example,
    # groups[0] = {"sigma": 0.1, "xi": 0.1, "ARI": 0.1, "SSE": 0.1}

    # data for data group i: data[10000*i: 10000*(i+1)], i=1, 2, 3, 4.
    # For example,
    # groups[i] = {"sigma": 0.1, "xi": 0.1, "ARI": 0.1, "SSE": 0.1}

    # groups is the dictionary above
    answers["cluster parameters"] = groups
    answers["1st group, SSE"] = {}

    # Create two scatter plots using `matplotlib.pyplot`` where the two
    # axes are the parameters used, with # \sigma on the horizontal axis
    # and \xi and the vertical axis. Color the points according to the SSE value
    # for the 1st plot and according to ARI in the second plot.

    # Choose the cluster with the largest value for ARI and plot it as a 2D scatter plot.
    # Do the same for the cluster with the smallest value of SSE.
    # All plots must have x and y labels, a title, and the grid overlay.



    # Plot is the return value of a call to plt.scatter()
    plot_ARI = plt.scatter([1,2,3], [4,5,6])
    plot_SSE = plt.scatter([1,2,3], [4,5,6])
    answers["cluster scatterplot with largest ARI"] = plot_ARI
    answers["cluster scatterplot with smallest SSE"] = plot_SSE

    # Pick the parameters that give the largest value of ARI, and apply these
    # parameters to datasets 1, 2, 3, and 4. Compute the ARI for each dataset.
    # Calculate mean and standard deviation of ARI for all five datasets.

    # A single float
    answers["mean_ARIs"] = mean_ari

    # A single float
    answers["std_ARIs"] = std_ari

    # A single float
    answers["mean_SSEs"] = mean_sse

    # A single float
    answers["std_SSEs"] = std_sse

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    all_answers = jarvis_patrick_clustering()
    with open("jarvis_patrick_clustering.pkl", "wb") as fd:
        pickle.dump(all_answers, fd, protocol=pickle.HIGHEST_PROTOCOL)
