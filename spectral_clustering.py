"""
Work with Spectral clustering.
Do not use global variables!
"""

import pickle
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian as csgraph_laplacian
from scipy.sparse.linalg import eigsh
from scipy.cluster.vq import kmeans, vq
from scipy.special import comb

######################################################################
#####     CHECK THE PARAMETERS     ########
######################################################################


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
    sigma = params_dict['sigma']
    n_clusters = params_dict['k']

    # Calculate the affinity matrix using the Gaussian kernel
    affinity_matrix = np.exp(-cdist(data, data) ** 2 / (2. * sigma ** 2))
    affinity_matrix_sparse = csr_matrix(affinity_matrix)

    # Compute the graph Laplacian matrix
    laplacian, diag = csgraph_laplacian(affinity_matrix_sparse, normed=False, return_diag=True)

    # Compute the first k eigenvectors of the graph Laplacian matrix
    eigenvalues, eigenvectors = eigsh(laplacian, k=n_clusters, which='SM')

    # Perform k-means clustering on the rows of the eigenvectors
    centroids, _ = kmeans(eigenvectors, n_clusters)
    computed_labels, _ = vq(eigenvectors, centroids)

    # Calculating SSE
    SSE = np.sum((data - centroids[computed_labels]) ** 2)

    # Calculating ARI
    ARI = calculate_ari(labels, computed_labels)

    return computed_labels, SSE, ARI, eigenvalues

def calculate_ari(labels_true, labels_pred):
    """
    Calculate the adjusted Rand index using a contingency matrix.
    """
    contingency_matrix = np.histogram2d(labels_true, labels_pred, bins=(np.max(labels_true)+1, np.max(labels_pred)+1))[0]
    sum_comb_c = np.sum([comb(n, 2) for n in np.sum(contingency_matrix, axis=1)])
    sum_comb_k = np.sum([comb(n, 2) for n in np.sum(contingency_matrix, axis=0)])
    sum_comb = np.sum([comb(n, 2) for n in contingency_matrix.flatten()])
    prod_comb = sum_comb_c * sum_comb_k / comb(len(labels_true), 2)
    ARI = (sum_comb - prod_comb) / ((sum_comb_c + sum_comb_k) / 2 - prod_comb)
    return ARI


def spectral_clustering():
    """
    Performs DENCLUE clustering on a dataset.

    Returns:
        answers (dict): A dictionary containing the clustering results.
    """

    answers = {}

    # Return your `spectral` function
    answers["spectral_function"] = spectral
    
    data = np.load('question2_cluster_data.npy')
    data_labels = np.load('question2_cluster_labels.npy')
    all_results = []
    no_of_batches = len(data) // 10000
    params_dict = {'sigma': 1.5, 'k': 5}
    for i in range(no_of_batches + 1):  # Adding 1 to handle the remainder if any
        start_index = 10000 * i
        end_index = 10000 * (i + 1)

        # Slice data and labels
        batch_data = data[start_index:end_index]
        batch_labels = data_labels[start_index:end_index]

        # Check if there's data to process
        if len(batch_data) > 0:
            # Call the spectral function and append results
            results = spectral(batch_data, batch_labels, params_dict)
            all_results.append(results)
            print(all_results)

    # Work with the first 10,000 data points: data[0:10000]
    # Do a parameter study of this data using Spectral clustering.
    # Minimmum of 10 pairs of parameters ('sigma' and 'xi').

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
