"""
Work with Spectral clustering.
Do not use global variables!
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import pickle
import scipy
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh
from scipy.cluster.vq import kmeans2, vq
from scipy.special import comb

######################################################################
#####     CHECK THE PARAMETERS     ########
######################################################################

def calculate_ari(labels_true, labels_pred):
    """
    Calculate the Adjusted Rand Index (ARI) given the true and predicted labels.
    
    Parameters:
    - labels_true : array-like, shape (n_samples,)
        True cluster labels
    - labels_pred : array-like, shape (n_samples,)
        Cluster labels predicted by the algorithm
    
    Returns:
    - ari : float
        The Adjusted Rand Index score
    """
    # Contingency table: Counting the number of common occurrences
    contingency_matrix = np.histogram2d(labels_true, labels_pred, bins=(np.unique(labels_true).size,
                                                                        np.unique(labels_pred).size))[0]

    # Sum over rows & columns
    sum_comb_c = np.sum([comb(n_c, 2) for n_c in np.sum(contingency_matrix, axis=1)])
    sum_comb_k = np.sum([comb(n_k, 2) for n_k in np.sum(contingency_matrix, axis=0)])

    # Sum over the whole matrix & calculate the combinatorial of all elements
    sum_comb = np.sum([comb(n_ij, 2) for n_ij in contingency_matrix.flatten()])
    comb_all = comb(np.sum(contingency_matrix), 2)

    # Compute the expected index (as per the ARI formula)
    expected_index = sum_comb_c * sum_comb_k / comb_all
    max_index = (sum_comb_c + sum_comb_k) / 2

    # Calculate ARI
    ari = (sum_comb - expected_index) / (max_index - expected_index)

    return ari

def spectral(
    data: NDArray[np.floating], labels: NDArray[np.int32], params_dict: dict
) -> tuple[
    NDArray[np.int32] | None, float | None, float | None, NDArray[np.floating] | None
]:
    """
    Implementation of the Spectral clustering  algorithm only using the `numpy` module.

    Arguments:
    - data: a set of points of shape 50,000 x 2.
    - dict: dictionary of parameters. The following two parameters must
       be present: 'sigma', and 'k'. There could be others.
       params_dict['sigma']:  in the range [.1, 10]
       params_dict['k']: the number of clusters, set to five.

    Return values:
    - computed_labels: computed cluster labels
    - SSE: float, sum of squared errors
    - ARI: float, adjusted Rand index
    - eigenvalues: eigenvalues of the Laplacian matrix
    """
    sigma = params_dict.get('sigma')
    N = params_dict.get('N')
    
    # Construct the affinity matrix using the Gaussian kernel
    dists = squareform(pdist(data, 'euclidean'))
    affinity_matrix = np.exp(-dists ** 2 / sigma)
    
    # Compute the Laplacian
    L = laplacian(affinity_matrix, normed=True)
    
    # Compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = eigsh(L, k=N+1, which='SM', tol=1e-10)
    eigenvectors = eigenvectors[:, 1:N+1]

    # Repeatedly run k-means until all clusters have at least one point
    max_attempts = 10
    attempt = 0
    while attempt < max_attempts:
        centroids, computed_labels = kmeans2(eigenvectors, k=N, minit='points')
        # Check if any cluster is empty
        if len(set(computed_labels)) == N:
            break
        attempt += 1

    # If after max_attempts there are still empty clusters, raise an error
    if len(set(computed_labels)) < N:
        raise ValueError("One of the clusters is empty after several initialization attempts.")

    # Compute SSE in the eigenvector space
    distances_to_centroids = cdist(eigenvectors, centroids)
    min_distances = np.min(distances_to_centroids, axis=1)
    SSE = np.sum(min_distances ** 2)
    
    # Compute ARI if ground truth labels are provided
    ARI = calculate_ari(labels, computed_labels)

    """
    computed_labels: NDArray[np.int32] | None = None
    SSE: float | None = None
    ARI: float | None = None
    eigenvalues: NDArray[np.floating] | None = None
    """

    return computed_labels, SSE, ARI, eigenvalues


def spectral_clustering():
    """
    Performs spectral clustering on a dataset.

    Returns:
        answers (dict): A dictionary containing the clustering results.
    """

    answers = {}

    # Return your `spectral` function
    answers["spectral_function"] = spectral

    # Work with the first 10,000 data points: data[0:10000]
    # Do a parameter study of this data using Spectral clustering.
    # Minimmum of 10 pairs of parameters ('sigma' and 'xi').

    data = np.load("question2_cluster_data.npy")
    data_labels = np.load("question2_cluster_labels.npy")
    params_dict = {'sigma' : 0.1, 'N' : 5}
    first_iter = spectral(data[0:1000], data_labels[0:1000], params_dict)
    print(first_iter)

    # Create a dictionary for each parameter pair ('sigma' and 'xi').
    groups = {}

    # For the spectral method, perform your calculations with 5 clusters.
    # In this cas,e there is only a single parameter, σ.

    # data for data group 0: data[0:10000]. For example,
    # groups[0] = {"sigma": 0.1, "ARI": 0.1, "SSE": 0.1}

    # data for data group i: data[10000*i: 10000*(i+1)], i=1, 2, 3, 4.
    # For example,
    # groups[i] = {"sigma": 0.1, "ARI": 0.1, "SSE": 0.1}

    # groups is the dictionary above
    answers["cluster parameters"] = groups
    answers["1st group, SSE"] = {}

    # Identify the cluster with the lowest value of ARI. This implies
    # that you set the cluster number to 5 when applying the spectral
    # algorithm.

    # Create two scatter plots using `matplotlib.pyplot`` where the two
    # axes are the parameters used, with \sigma on the horizontal axis
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

    # Plot of the eigenvalues (smallest to largest) as a line plot.
    # Use the plt.plot() function. Make sure to include a title, axis labels, and a grid.
    plot_eig = plt.plot([1,2,3], [4,5,6])
    answers["eigenvalue plot"] = plot_eig

    # Pick the parameters that give the largest value of ARI, and apply these
    # parameters to datasets 1, 2, 3, and 4. Compute the ARI for each dataset.
    # Calculate mean and standard deviation of ARI for all five datasets.

    # A single float
    answers["mean_ARIs"] = 0.

    # A single float
    answers["std_ARIs"] = 0.

    # A single float
    answers["mean_SSEs"] = 0.

    # A single float
    answers["std_SSEs"] = 0.

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    all_answers = spectral_clustering()
    with open("spectral_clustering.pkl", "wb") as fd:
        pickle.dump(all_answers, fd, protocol=pickle.HIGHEST_PROTOCOL)
