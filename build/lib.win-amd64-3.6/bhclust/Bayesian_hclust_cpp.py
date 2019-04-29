import numpy as np
import itertools
from scipy.special import gammaln



class bayesian_hclust_cpp():
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
        while num_nodes > 1:
            #print(i,)
            max_posterior = float('-Inf')
            merge_node = None
            
            for lnode_inx, rnode_inx in itertools.combinations(data_nodes.keys(), 2):
                # try each pair of current clusters, calculate the posterior prob of merged node
                merge_node_temp = node.merge(data_nodes[lnode_inx], data_nodes[rnode_inx], self.model)
                
                log_posterior = merge_node_temp.log_pik +\
                                self.model(merge_node_temp.data, ) -\
                                merge_node_temp.log_marginal
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

    

class node():
    """
    Hierarchical clustering node with prior/posterior probabilities
    
    Attributes:
    -----------------------------
    data: numpy.ndarray
        data in the node, (n_k, p)
    model: class
        model to calculate marginal likelihood
    alpha: float
        concentration parameter in Dirichlet Process Mixture model
    log_dk: float
        log of d_k
    log_pik: float
        log of pi_k, i.e. log of prior
    nk: int
        number of data in the nodes
    log_marginal: float
        log of marginal probability of the data in the node
    """
    def __init__(self, data, model, alpha, log_dk, log_pik=0., log_marginal=None):
        """
        Parameters:
        ----------------------------------------
        data: numpy.ndarray
            data in the node, (n_k, p)
        model: class
            model to calculate marginal likelihood
        alpha: float
            concentration parameter in Dirichlet Process Mixture model
        log_dk: float
            log of d_k
        log_pik: float
            log of pi_k, i.e. log of prior
        """
        
        self.data = np.array(data)
        self.model = model
        self.alpha = alpha
        self.nk = data.shape[0]
        self.log_dk = log_dk
        self.log_pik = log_pik
        if log_marginal is None:
            self.log_marginal = self.model(self.data)
        else:
            self.log_marginal = log_marginal
    
    @classmethod
    def merge(cls, lnode, rnode, model):
        """
        Merge two nodes.
        
        Parameters:
        ------------------
        lnode: Node object
            left node
        rnode: Node object
            right node
        
        Returns:
        ------------------
        Node object
            the new merged node object
        """
        
        new_data = np.vstack((lnode.data, rnode.data))
        nk = new_data.shape[0]
        alpha = lnode.alpha
        
        # use numpy.logaddexp to avoid precision issues for small floating number
        log_dk = np.logaddexp(np.log(alpha)+gammaln(nk), lnode.log_dk+rnode.log_dk)
        log_pik = np.log(alpha) + gammaln(nk) - log_dk
        
        log_lkh_left = lnode.log_marginal
        log_lkh_right = rnode.log_marginal
        log_lkh_merge = model(new_data)

        log_marginal = np.logaddexp(log_pik + log_lkh_merge, 
                                    np.log(-np.expm1(log_pik)) + log_lkh_left + log_lkh_right)
        
        # return a new node
        return cls(new_data, model, alpha, log_dk, log_pik, log_marginal)
        
        
    @staticmethod
    def distance(lnode, rnode, linkage='average'):
        """
        Calculate distance of two clusters using linkage
        
        Parameters:
        ---------------
        lnode: Node object
            left node
        rnode: Node object
            right node
        linkage: string
            specify linkage method
        
        Returns:
        ---------------
        distance: float
        """
        
        if linkage == "average":
            lcenter = np.mean(lnode.data, axis=0)
            rcenter = np.mean(rnode.data, axis=0)
            distance = np.sqrt(np.sum((lcenter - rcenter) ** 2))
        
        return distance