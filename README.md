A novel CVI: Distance-based Separability Index (DSI)
=============
This is an example of computing Cluster Validity Index (CVI) on datasets after clustering.

It includes codes to compute the distance-based separability index (DSI).

Related paper
=============
### An Internal Cluster Validity Index Using a Distance-based Separability Measure

`Citation` S. Guan and M. Loew, "An Internal Cluster Validity Index Using a Distance-based Separability Measure," 2020 IEEE 32nd International Conference on Tools with Artificial Intelligence (ICTAI), Baltimore, MD, USA, 2020, pp. 827-834, doi: 10.1109/ICTAI50040.2020.00131.

[`Arxiv`] (https://arxiv.org/abs/2009.01328)

Results
=============
### DSI scores of the [handwritten digits dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html) after applying several clustering methods
Output in `results.txt`.

* 0.4933647148073444  `ture labels`
* 0.4836361123766773  `K-Means`
* 0.5716958115555282  `Spectral Clustering`
* 0.5471394199023191  `Birch`
* 0.4697710019443232  `Gaussian Mixture (EM)`
