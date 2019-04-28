This is the final project of sta 663.

The repository contains the implementations and tests of Bayesian Hierarchical Clustering for multivariate Gaussian and Bernoulli data.

## Installation
1. Clone the git repo by

```sh
git clone https://github.com/ulandddda/Bayesian-Hierarchical-Clustering.git
```

2. Go to the local repo, type

```sh
python3 setup.py install build_ext -i
```


## Usage

The test code and examples are in test/test_bhc.ipynb.

```python
from bhclust.pytest_wrapper import *
np.random.seed(8)
simu_data = generate_data()
Z, rk, clusters_ = fit_bhc_cpp(simu_data, cutoff=3)
draw_dendrogram(Z, thres=0.5)
```

## Reference
Heller, K. A., & Ghahramani, Z. (2005, August). Bayesian hierarchical clustering. In Proceedings of the 22nd international conference on Machine learning (pp. 297-304). ACM.

https://archive.ics.uci.edu/ml/datasets/glass+identification 

https://archive.ics.uci.edu/ml/datasets/glass+identification
