"""
Work with Jarvis-Patrick clustering.
Do not use global variables!
"""
import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import pickle

######################################################################
#####     CHECK THE PARAMETERS     ########
######################################################################

def jarvis_patrick(
    data: NDArray[np.floating], labels: NDArray[np.int32], params_dict: dict
) -> tuple[NDArray[np.int32] | None, float | None, float | None]:
    """
    Implementation of the Jarvis-Patrick algorithm only using the `numpy` module.

    Arguments:
    - data: a set of points of shape 50,000 x 2.
    - dict: dictionary of parameters. The following two parameters must
       be present: 'k', 'smin', There could be others.
    - params_dict['k']: the number of nearest neighbours to consider. This determines the size of the neighborhood used to assess the similarity between datapoints.
    Choose values in the range 3 to 8
    - params_dict['smin']:  the minimum number of shared neighbours to consider two points in the same cluster.
       Choose values in the range 4 to 10.

    Return values:
    - computed_labels: computed cluster labels
    - SSE: float, sum of squared errors
    - ARI: float, adjusted Rand index

    Notes:
    - the nearest neighb can be bidirectional or unidirectional
    - Bidirectional: if point A is a nearest neighbor of point B, then point B is a nearest neighbor of point A).
    - Unidirectional: if point A is a nearest neighbor of point B, then point B is not necessarily a nearest neighbor of point A).
    - In this project, only consider unidirectional nearest neighboars for simplicity.
    - The metric  used to compute the the k-nearest neighberhood of all points is the Euclidean metric
    """
    labels_tr=labels
    def cs_nm(data, k, t):
        mat_dist = squareform(pdist(data, 'euclidean'))
        neighb = np.argsort(mat_dist, axis=1)[:, 1:k+1]
        n = len(data)
        mat_adj = np.zeros((n, n), dtype=bool)

        for i in range(n):
            for j in range(i + 1, n):
                shared_neighb = len(set(neighb[i]).intersection(neighb[j]))
                if shared_neighb >= t:
                    mat_adj[i, j] = True
                    mat_adj[j, i] = True

        return mat_adj

    def calculate_sse(data, labels, cst_cc):
        sse = 0
        for k in range(len(cst_cc)):
            cluster_data = data[labels == k]
            sse += np.sum((cluster_data - cst_cc[k])**2)
        return sse
    
    def adjusted_rand_index(labels_true, labels_pred):
        classes = np.unique(labels_true)
        clusters = np.unique(labels_pred)

        tab_cont = np.zeros((classes.size, clusters.size), dtype=int)
        for class_idx, lab_cls in enumerate(classes):
            for id_cstx, lab_clst in enumerate(clusters):
                tab_cont[class_idx, id_cstx] = np.sum((labels_true == lab_cls) & (labels_pred == lab_clst))

        sum_over_rows = np.sum(tab_cont, axis=1)
        sum_over_cols = np.sum(tab_cont, axis=0)

        n_comb = sum([n_ij * (n_ij - 1) / 2 for n_ij in tab_cont.flatten()])
        sum_over_rows_comb = sum([n_ij * (n_ij - 1) / 2 for n_ij in sum_over_rows])
        sum_over_cols_comb = sum([n_ij * (n_ij - 1) / 2 for n_ij in sum_over_cols])

        n = labels_true.size
        total_combinations = n * (n - 1) / 2
        expected_index = sum_over_rows_comb * sum_over_cols_comb / total_combinations
        max_index = (sum_over_rows_comb + sum_over_cols_comb) / 2
        denmntr = (max_index - expected_index)

        if denmntr == 0:
            return 1 if n_comb == expected_index else 0
        ari = (n_comb - expected_index) / denmntr

        return ari
    
    def cust_db(matrix, data, minPts):
        n = matrix.shape[0]
        labels = -np.ones(n)
        id_cst = 0
        cst_cc = []

        for i in range(n):
            if labels[i] != -1:
                continue
            neighb = np.where(matrix[i])[0]
            if len(neighb) < minPts:
                labels[i] = -2
                continue
            labels[i] = id_cst
            seed_set = set(neighb)

            pnt_clst = [data[i]]

            while seed_set:
                pt_crrnt = seed_set.pop()
                pnt_clst.append(data[pt_crrnt])
                if labels[pt_crrnt] == -2:
                    labels[pt_crrnt] = id_cst
                if labels[pt_crrnt] != -1:
                    continue
                labels[pt_crrnt] = id_cst
                current_neighb = np.where(matrix[pt_crrnt])[0]
                if len(current_neighb) >= minPts:
                    seed_set.update(current_neighb)

            cluster_center = np.mean(pnt_clst, axis=0)
            cst_cc.append(cluster_center)
            id_cst += 1

        return labels, np.array(cst_cc)

    mat_adj = cs_nm(data, k=params_dict['k'], t=2)
    labels, cst_cc = cust_db(mat_adj, data, minPts=params_dict['smin'])
    sse = calculate_sse(data, labels, cst_cc)

    ari = adjusted_rand_index(labels_tr, labels)
    computed_labels: NDArray[np.int32] | None = labels
    SSE: float | None = sse
    ARI: float | None = ari

    return computed_labels, SSE, ARI


def jarvis_patrick_clustering():
    """
    Performs Jarvis-Patrick clustering on a dataset.

    Returns:
        answers (dict): A dictionary containing the clustering results.
    """


    answers = {}
    data=np.load("question1_cluster_data.npy")
    labels_tr=np.load("question1_cluster_labels.npy")
    # Return your `spectral` function
    answers["jarvis_patrick_function"] = jarvis_patrick

    # Work with the first 10,000 data points: data[0:10000]
    # Do a parameter study of this data using Spectral clustering.
    # Minimmum of 10 pairs of parameters ('sigma' and 'xi').

    # Create a dictionary for each parameter pair ('sigma' and 'xi').
    groups = {}

    # For the spectral method, perform your calculations with 5 clusters.
    # In this cas,e there is only a single parameter, σ.
    """
    # Find best k by testing each k value independently ##Result in 5
    for k in k_values:
        computed_labels, SSE, ARI = jarvis_patrick(data_samples, label_samples, {'k': k, 'smin': 6})  # Fixed smin for testing k
        print(f"Testing k={k}, ARI={ARI}")
        if ARI > best_ari:
            best_ari = ARI
            best_k = k
    print(f"Best k value on the initial slice: {best_k} with ARI: {best_ari}")

    # Reset best_ari to find best smin
    best_ari = -1

    # Find best smin by testing each smin value independently ##Result is 4
    for smin in smin_values:
        computed_labels, SSE, ARI = jarvis_patrick(data_samples, label_samples, {'k': best_k, 'smin': smin})
        print(f"Testing smin={smin}, ARI={ARI}")
        if ARI > best_ari:
            best_ari = ARI
            best_smin = smin
    print(f"Best smin value on the initial slice: {best_smin} with ARI: {best_ari}")

    """
    fin_sse=[]
    fin_prd=[]
    fin_ari=[]
    num_samples = 1000
    for i in range(5):
      data_int = data[i*1000:(i+1)*1000]
      tr_labels = labels_tr[i*1000:(i+1)*1000]
      params_dict={'k':5,'smin':5}
      preds,sse_hyp,ari_hyp,=jarvis_patrick(data_int,tr_labels,params_dict)
      fin_sse.append(sse_hyp)
      fin_ari.append(ari_hyp)
      fin_prd.append(preds)
      if i not in groups:
        groups[i]={'k':5,'smin':5,'ARI':ari_hyp,"SSE":sse_hyp}
      else:
        pass

    sse_numpy=np.array(fin_sse)
    ari_numpy=np.array(fin_ari)

    answers["cluster parameters"] = groups
    answers["1st group, SSE"] = groups[0]['SSE']

    # Create two scatter plots using `matplotlib.pyplot`` where the two
    # axes are the parameters used, with \sigma on the horizontal axis
    # and \xi and the vertical axis. Color the points according to the SSE value
    # for the 1st plot and according to ARI in the second plot.
    least_sse_index=np.argmin(sse_numpy)
    highest_ari_index=np.argmax(ari_numpy)

    plt.figure(figsize=(10, 8))
    plot_ARI=plt.scatter(data[1000*highest_ari_index:(highest_ari_index+1)*1000, 0], data[1000*highest_ari_index:(highest_ari_index+1)*1000, 1], c=fin_prd[highest_ari_index], cmap='viridis', marker='.')
    plt.title(f"Question 1 - Jarvis Patrick - Largest ARI")
    plt.xlabel(f'Feature 1 for Dataset{i+1}')
    plt.ylabel(f'Feature 2 for Dataset{i+1}')
    plt.grid(True)
    cbar = plt.colorbar(plot_ARI, label='Cluster Label')
    plt.savefig("Question_1_Jarvis_Patrick_Clustering_Result_with_Largest_ARI.pdf")
    plt.show()

    plt.figure(figsize=(10, 8))
    plot_SSE=plt.scatter(data[1000*least_sse_index:(least_sse_index+1)*1000, 0], data[1000*least_sse_index:(least_sse_index+1)*1000, 1], c=fin_prd[least_sse_index], cmap='viridis', marker='.')
    plt.xlabel(f'Feature 1 for Dataset{i+1}')
    plt.ylabel(f'Feature 2 for Dataset{i+1}')
    plt.title(f"Question 1 - Jarvis Patrick - Smallest SSE")
    plt.grid(True)
    cbar1 = plt.colorbar(plot_SSE, label='Cluster Label')
    plt.savefig("Question_1_Jarvis_Patrick_Clustering_Result_with_smallest_SSE.pdf")
    plt.show()
    answers["cluster scatterplot with largest ARI"] = plot_ARI
    answers["cluster scatterplot with smallest SSE"] = plot_SSE

    # # Plot of the eigenvalues (smallest to largest) as a line plot.
    # # Use the plt.plot() function. Make sure to include a title, axis labels, and a grid.

    # Pick the parameters that give the largest value of ARI, and apply these
    # parameters to datasets 1, 2, 3, and 4. Compute the ARI for each dataset.
    # Calculate mean and standard deviation of ARI for all five datasets.
    ARI_sum=[]
    SSE_sum=[]
    for i in groups:
      if 'ARI' in groups[i]:
        ARI_sum.append(groups[i]['ARI'])
        SSE_sum.append(groups[i]['SSE'])

    # A single float
    answers["mean_ARIs"] = float(np.mean(ari_numpy))

    # A single float
    answers["std_ARIs"] = float(np.std(ari_numpy))

    # A single float
    answers["mean_SSEs"] = float(np.mean(sse_numpy))

    # A single float
    answers["std_SSEs"] = float(np.std(sse_numpy))

    return answers

# ----------------------------------------------------------------------
if __name__ == "__main__":
    all_answers = jarvis_patrick_clustering()
    with open("jarvis_patrick_clustering.pkl", "wb") as fd:
        pickle.dump(all_answers, fd, protocol=pickle.HIGHEST_PROTOCOL)
