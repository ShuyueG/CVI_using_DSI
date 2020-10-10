'''
This is an example of computing CVIs on datasets after clustering.
It includes codes to compute the distance-based separability index (DSI).

Datasets:           Wine recognition dataset (can add more)
CVIs:               DSI (can add more)
Clustering Methods: KMeans
                    Spectral Clustering
                    BIRCH
                    GaussianMixture (EM)
                    (can add more)

Related paper:      An Internal Cluster Validity Index Using a Distance-based Separability Measure
                    [In press] International Conference on Tools with Artificial Intelligence (ICTAI), 2020
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

np.random.seed(0)

# ==================================================
# Datasets
# ==================================================

# load dataset ###################################
wine = skdata.load_wine(return_X_y=True)

# put in dataset list ###################################
datasets = [
    # ('name',dataset,{'n_clusters': # of clusters}),
    ('wine', wine, {'n_clusters': 3})
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

# put in CVI list ###################################
measures = [
    # ('name',function),
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
                if meas_name == 'DSI':
                    score = method(X, y_pred)
            except:
                score = '-'  # if error

            with open('results.txt', 'a') as fw:  # record CVIs
                fw.write('\t' + str(score))
