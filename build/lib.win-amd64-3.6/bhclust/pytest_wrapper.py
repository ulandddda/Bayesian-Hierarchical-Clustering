from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
import cppimport
import pandas as pd

from bhclust.Bayesian_hclust import *
from bhclust.Bayesian_hclust_cpp import *
from bhclust.Bayesian_hclust_cpp_fast import *
from bhclust.Bayesian_hclust_fast import *
from log_marginal_prob import *


def generate_data(num_per_cluster = [20, 20, 20], 
                  center = [[-5., -1.], [0., 2.], [5., 3.]], 
                  cov = [[.3, 0], [0, .3]]):
    n_clusters = len(num_per_cluster)
    X = np.zeros((1, 2))
    for i in range(n_clusters):
        X_temp = np.random.multivariate_normal(center[i], cov, num_per_cluster[i])
        X = np.vstack((X, X_temp))
    return X[1:,]

def fit_bhc(data, cutoff=3, m=np.zeros(2), S=np.eye(2)*0.3, r=1.0, v=3.0, alpha=1.0):
    normal_data_model = normal_inversewishart(m, S, r, v)
    bhclust = bayesian_hclust(model=normal_data_model, alpha=alpha)
    Z, rk, clusters_ = bhclust.fit(data, cutoff)
    return Z, rk, clusters_

def fit_bhc_cpp(data, cutoff=3, m=np.zeros(2), S=np.eye(2)*0.3, r=1.0, v=3.0, alpha=1.0):
    normal_model = partial(log_marginal_probability_cpp, m, S, r, v)
    bhclust = bayesian_hclust_cpp(model=normal_model, alpha=alpha)
    Z, rk, clusters_ = bhclust.fit(data, cutoff)
    return Z, rk, clusters_

def fit_bhc_cpp_fast(data, cutoff=3, m=np.zeros(2), S=np.eye(2)*0.3, r=1.0, v=3.0, alpha=1.0):
    normal_model = partial(log_marginal_probability_cpp, m, S, r, v)
    bhclust = bayesian_hclust_cpp_fast(model=normal_model, alpha=alpha)
    Z, rk, clusters_ = bhclust.fit(data, cutoff)
    return Z, rk, clusters_

def fit_bhc_fast(data, cutoff=3, m=np.zeros(2), S=np.eye(2)*0.3, r=1.0, v=3.0, alpha=1.0):
    normal_data_model = normal_inversewishart(m, S, r, v)
    bhclust = bayesian_hclust_fast(model=normal_data_model, alpha=alpha)
    Z, rk, clusters_ = bhclust.fit(data, cutoff)
    return Z, rk, clusters_

def fit_bhc_benoulli(data, cutoff, alpha, beta, alpha_=1.0):
    bernoulli_data_model = bernoulli_beta(alpha, beta)
    bhclust = bayesian_hclust(model=bernoulli_data_model, alpha=alpha_)
    Z, rk, clusters_ = bhclust.fit(data, cutoff)
    return Z, rk, clusters_
	
def process_data(filename, n_samples=[10, 10], labels=[1, 2]):
    data = pd.read_table(filename, delimiter = ',', header=None)
    data.dropna(inplace=True)
    data_1 = data.loc[data.iloc[:,-1]==labels[0], :]
    data_2 = data.loc[data.iloc[:,-1]==labels[1], :]
    np.random.seed(16)
    data_1_sample = data_1.sample(n=n_samples[0])
    data_1_sample.iloc[:, -1] = 1
    data_2_sample = data_2.sample(n=n_samples[1])
    data_2_sample.iloc[:, -1] = 0
    sample = pd.concat([data_1_sample, data_2_sample])
    sample_feature = sample.iloc[:,:-1]
    return sample, sample_feature
	
def draw_dendrogram(Z, fig_w = 12, fig_h = 6, thres = 0.5):
    plt.figure(figsize=(12,6))
    dendrogram(Z, color_threshold=thres*max(Z[:,2]))
    pass

def draw_scatter(clusters_):
    for k in clusters_.keys():
        plt.scatter(clusters_[k].data[:, 0], clusters_[k].data[:, 1])
    pass
	
def calculate_purity(cluster, data_all):
    n = data_all.shape[0]
    correct = []
    p = 0
    for k in cluster.keys():
        cluster_data = cluster[k].data
        n_cluster = cluster_data.shape[0]
        #print(n_cluster)
        for i in range(n_cluster):
            for j in range(n):
                if np.allclose(cluster_data[i,:], data_all[j,:-1]):
                     correct.append(p^int(data_all[j,-1]))
        p = 1
    return max(sum(correct) / n, 1 - sum(correct) / n)