A novel CVI: Distance-based Separability Index (DSI)
=============
This is an example of computing Cluster Validity Index (CVI) on datasets after clustering.

It includes codes to compute the distance-based separability index (DSI).

Related paper
=============
### An Internal Cluster Validity Index Using a Distance-based Separability Measure

`In press` International Conference on Tools with Artificial Intelligence (ICTAI), 2020

https://arxiv.org/abs/2009.01328

Results
=============
### DSI scores of the [wine recognition dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html) after applying several clustering methods
Output in `results.txt`.

* 0.6449753215024018  `ture labels`
* 0.6315623258366395  `K-Means`
* 0.6292599916034574  `Spectral Clustering`
* 0.6086220213650594  `Birch`
* 0.6328543138866538  `Gaussian Mixture (EM)`
