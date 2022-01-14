'''
This is an example of computing CVIs on datasets after clustering.
It includes codes to compute the distance-based separability index (DSI).

Datasets:           Optical recognition of handwritten digits dataset
                    (can add more)
                    
CVIs:               DSI 
                    Dunn
                    CH
                    DB
                    Silhouette
                    (can add more)
                    
Clustering Methods: KMeans
                    Spectral Clustering
                    BIRCH
                    GaussianMixture (EM)
                    (can add more)

Related paper:      An Internal Cluster Validity Index Using a Distance-based Separability Measure
                    International Conference on Tools with Artificial Intelligence (ICTAI), 2020
                    https://arxiv.org/abs/2009.01328

By:                 Shuyue Guan
                    https://shuyueg.github.io/
'''

import numpy as np
import scipy.spatial.distance as distance
import sklearn.datasets as skdata
from scipy.stats import ks_2samp
from sklearn import cluster, mixture
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics.pairwise import euclidean_distances

np.random.seed(0)

# ==================================================
# Datasets
# ==================================================

# load dataset ###################################
digits = skdata.load_digits(n_class=10,return_X_y=True)

# put in dataset list ###################################
datasets = [
    # ('name',dataset,{'n_clusters': # of clusters}),
    ('digits', digits, {'n_clusters': 10})
]


# ==================================================
# CVIs
# ==================================================

# load or define CVIs ###################################
#####################  DSI ##################{
def dists(data, dist_func=distance.euclidean):  # compute ICD
    num = data.shape[0]
    data = data.reshape((num, -1))
    dist = []
    for i in range(0, num - 1):
        for j in range(i + 1, num):
            dist.append(dist_func(data[i], data[j]))
    return np.array(dist)


def dist_btw(a, b, dist_func=distance.euclidean):  # compute BCD
    a = a.reshape((a.shape[0], -1))
    b = b.reshape((b.shape[0], -1))
    dist = []
    for i in range(a.shape[0]):
        for j in range(b.shape[0]):
            dist.append(dist_func(a[i], b[j]))
    return np.array(dist)


def separability_index_ks_2samp(X, labels):  # KS test on ICD and BCD
    classes = np.unique(labels)
    SUM = 0
    for c in classes:
        pos = X[np.squeeze(labels == c)]
        neg = X[np.squeeze(labels != c)]

        dist_pos = dists(pos)
        distbtw = dist_btw(pos, neg)
        D, _ = ks_2samp(dist_pos, distbtw)  # KS test
        SUM += D
    SUM = SUM / classes.shape[0]  # normed: b/c ks_2samp ranges [0,1]
    return SUM


#####################################################}

##################### Dunn  ##################{
""" AUTHOR: "Joaquim Viegas"
JQM_CV - Python implementations of Dunn and Davis Bouldin clustering validity indices

dunn_fast(points, labels):
    Fast implementation of Dunn index that depends on numpy and sklearn.pairwise
    -- No Cython implementation
"""

def dunn_fast(points, labels):
    """ Dunn index - FAST (using sklearn pairwise euclidean_distance function)

    Parameters
    ----------
    points : np.array
        np.array([N, p]) of all points
    labels: np.array
        np.array([N]) labels of all points
    """
    distances = euclidean_distances(points)
    ks = np.sort(np.unique(labels))

    deltas = np.ones([len(ks), len(ks)]) * 1000000
    big_deltas = np.zeros([len(ks), 1])

    l_range = list(range(0, len(ks)))

    for k in l_range:
        for l in (l_range[0:k] + l_range[k + 1:]):
            deltas[k, l] = delta_fast((labels == ks[k]), (labels == ks[l]), distances)

        big_deltas[k] = big_delta_fast((labels == ks[k]), distances)

    di = np.min(deltas) / np.max(big_deltas)
    return di
#####################################################}

# put in CVI list ###################################
measures = [
    ## ('name',function),
    # ('Dunn', dunn_fast),
    # ('CH', metrics.calinski_harabasz_score),
    # ('DB', metrics.davies_bouldin_score),
    # ('Silhouette', metrics.silhouette_score),
    # ('ARI', metrics.adjusted_rand_score),
    ('DSI', separability_index_ks_2samp)
]


# ==================================================
# Main Process
# ==================================================

#########################################################################
for i_dataset, dataset, algo_params in datasets:  # loop datasets

    # default setting for clustering ###################################
    default_base = {'quantile': .3,
                    'eps': .3,
                    'damping': .9,
                    'preference': -200,
                    'n_neighbors': 10,
                    'n_clusters': 3,
                    'min_samples': 20,
                    'xi': 0.05,
                    'min_cluster_size': 0.1}

    # update parameters with dataset-specific values
    params = default_base.copy()
    params.update(algo_params)

    X, y = dataset
    y = np.squeeze(y)

    # normalize dataset for easier parameter selection
    try:
        X = StandardScaler().fit_transform(X)
    except:
        X = StandardScaler(with_mean=False).fit_transform(X)


    # ==================================================
    # Clustering Methods
    # ==================================================

    # load or define clustering methods ###################################
    two_means = cluster.MiniBatchKMeans(n_clusters=params['n_clusters'])

    spectral = cluster.SpectralClustering(
        n_clusters=params['n_clusters'], eigen_solver='arpack',
        affinity="nearest_neighbors")

    birch = cluster.Birch(n_clusters=params['n_clusters'])

    gmm = mixture.GaussianMixture(
        n_components=params['n_clusters'], covariance_type='full')

    # put in clustering method list ###################################
    clustering_algorithms = [
        # ('name', method),
        ('REAL', {}),  # use true labels
        ('KMeans', two_means),
        ('SpectralClustering', spectral),
        ('Birch', birch),
        ('GaussianMixture', gmm)
    ]

    #########################################################################
    for name, algorithm in clustering_algorithms:  # loop clustering methods

        if name == 'REAL':  # use true labels as prediction
            y_pred = y
        else:

            algorithm.fit(X)  # apply

            if hasattr(algorithm, 'labels_'):
                y_pred = algorithm.labels_.astype(np.int)
            else:
                y_pred = algorithm.predict(X)

        with open('results.txt', 'a') as fw:  # new line for the next clustering method
            fw.write('\n')

        #########################################################################
        for meas_name, method in measures:  # loop CVIs
            try:
                if meas_name == 'Silhouette':
                    score = method(X, y_pred, metric='euclidean')
                elif meas_name == 'DSI':
                    score = method(X, y_pred, distance.euclidean)
                elif meas_name == 'ARI':
                    score = method(y, y_pred)
                else:
                    score = method(X, y_pred)
                    
            except:
                score = '-'  # if error

            with open('results.txt', 'a') as fw:  # record CVIs
                fw.write('\t' + str(score))
