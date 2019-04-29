import numpy as np
import itertools
from scipy.special import gammaln
from bhclust.Bayesian_hclust_cpp import node



class bayesian_hclust_cpp_fast():
    """
    Bayesian Hierarchical Clustering with critical function using c++ optimized
    """
    
    def __init__(self, model, alpha):
        """
        Parameters:
        ------------
        model: class
            model to calculate marginal likelihood
        alpha: float
            concentration parameter in Dirichlet Process Mixture model 
        """
        
        self.model = model
        self.alpha = alpha
    
    
    def fit(self, X, cutoff = 1):
        """
        Fit the tree and calculate marginal likelihood of each node
        Parameters:
        -------------
        X: numpy.ndarray
            data to be clustered
        cutoff: int
            cut off the tree so that in total `cutoff` clusters formed
        
        Returns:
        -------------
        Z: numpy.ndarray
            linkage matrix
        posterior: numpy.ndarray
            posterior (rk) at each cut
        clusters_: dictionary {int: node}
            clusters with nodes in
        """
        
        # init log_dk as alpha
        log_dk = np.log(self.alpha)    
        # init nodes dict
        data_nodes = dict((inx, node(np.array([row]), self.model, self.alpha, log_dk))
                         for inx, row in enumerate(X))
        
        num_nodes = len(data_nodes)    # modifiable
        n = len(data_nodes)    # unmodifiable
        # init linkage matrix
        Z = np.zeros((num_nodes-1, 4))
        posterior = np.zeros(num_nodes-1)
        # use to index linkage matrix Z
        i = 0
        rk_dict = {}
        node_dict = {}
        while num_nodes > 1:
            #print(i,)
            max_posterior = float('-Inf')
            merge_node = None
            
            for lnode_inx, rnode_inx in itertools.combinations(data_nodes.keys(), 2):
                # try each pair of current clusters, calculate the posterior prob of merged node
                if (lnode_inx, rnode_inx) in rk_dict.keys():
                    log_posterior = rk_dict[(lnode_inx, rnode_inx)]
                    merge_node_temp = node_dict[(lnode_inx, rnode_inx)]
                    #if i==9:
                     #   print(lnode_inx, rnode_inx, log_posterior)
                else:
                    merge_node_temp = node.merge(data_nodes[lnode_inx], data_nodes[rnode_inx], self.model)

                    log_posterior = merge_node_temp.log_pik +\
                                    self.model(merge_node_temp.data, ) -\
                                    merge_node_temp.log_marginal
                    #if i==9:
                     #   print("---", lnode_inx, rnode_inx, log_posterior)
                    rk_dict[(lnode_inx, rnode_inx)] = log_posterior
                    node_dict[(lnode_inx, rnode_inx)] = merge_node_temp
                    
                #print(log_posterior)
                if log_posterior > max_posterior:
                    max_posterior = log_posterior
                    merge_node = merge_node_temp
                    merge_left_inx = lnode_inx
                    merge_right_inx = rnode_inx
            
            # construct linkage matrix
            Z[i, 0] = merge_left_inx
            Z[i, 1] = merge_right_inx
            posterior[i] = np.exp(max_posterior)
            Z[i, 2] = node.distance(data_nodes[merge_right_inx], data_nodes[merge_left_inx], linkage="average")
            #Z[i, 2] = 1 - posterior[i]
            Z[i, 3] = merge_node.nk
            
            # return clusters formed at a specific cutoff position
            if num_nodes == cutoff:
                clusters_ = data_nodes.copy()
                
            del data_nodes[merge_right_inx]
            del data_nodes[merge_left_inx]
            data_nodes[n+i] = merge_node
            
            i += 1
            num_nodes -= 1

        return Z, posterior, clusters_
