
from pickle import TRUE
from re import I
import networkx as nx
import numpy as np
import grakel as gk
import warnings
from scipy.stats import norm


import pandas as pd
#import time


from numba import njit
from scipy.sparse.sputils import validateaxis

import MONK
import tqdm

# Biased empirical maximum mean discrepancy
def MMD_b(K: np.array, n1: int, n2: int):
    '''
    Biased empirical maximum mean discrepancy

    Parameters
    ---------------------
    K: np.array,
        Kernel matrix size K (n+m) x (n+m)
    n1: int,
        Number of observations in sample 1

    n2: int,
        Number of observations in sample 1

    Returns
    ----------------------------
    float
        Unbiased estimate of the maximum mean discrepancy
    
    '''

    if (n1 + n2) != K.shape[0]:
        raise ValueError("n + m have to equal the size of K")

    Kx = K[:n1, :n1]
    Ky = K[n1:, n1:]
    Kxy = K[:n1, n1:]
    
    # important to write 1.0 and not 1 to make sure the outcome is a float!
    return 1.0 / (n1 ** 2) * Kx.sum() + 1.0 / (n1 * n2) * Ky.sum() - 2.0 / (n2 ** 2) * Kxy.sum()

# Unbiased empirical maximum mean discrepancy
def MMD_u(K: np.array, n1: int, n2: int):
    '''
    Unbiased empirical maximum mean discrepancy

    Parameters
    ---------------------
    K: np.array,
        Kernel matrix size K (n+m) x (n+m)

    n1: int,
        Number of observations in sample 1

    n2: int,
        Number of observations in sample 1

    Returns
    ----------------------------
    float
        Unbiased estimate of the maximum mean discrepancy
    
    '''

    if (n1 + n2) != K.shape[0]:
        raise ValueError("n + m have to equal the size of K")
    Kx = K[:n1, :n1]
    Ky = K[n1:, n1:]
    Kxy = K[:n1, n1:]
    # important to write 1.0 and not 1 to make sure the outcome is a float!
    return 1.0 / (n1* (n1 - 1.0)) * (Kx.sum() - np.diag(Kx).sum()) + 1.0 / (n2 * (n2 - 1.0)) * (Ky.sum() - np.diag(Ky).sum()) - 2.0 / (n1 * n2) * Kxy.sum()

def MONK_EST(K, Q, n1, n2):
    """
    Wrapper for MONK class

    Parameters
    ---------------------
    K: np.array,
        Kernel matrix size K (n+m) x (n+m)

    Q: int,
        Number of partitions for each sample.

    n1: int,
        Number of observations in sample 1

    n2: int,
        Number of observations in sample 1

    Returns
    ----------------------------
    float
        Unbiased estimate of the maximum mean discrepancy


    """

    mmd =  MONK.MY_MMD_MOM(Q = Q,  K=K)
    return mmd.estimate(n1, n2)

def MMD_l(K: np.array, n1: int, n2: int) -> float:
    '''
    Unbiased estimate of the maximum mean discrepancy using fewer Kernel evaluations

    Parameters
    ---------------------
    K: np.array,
        Kernel matrix size K (n+m) x (n+m)
    n1: int,
        Number of observations in sample 1

    n2: int,
        Number of observations in sample 1

    Returns
    ----------------------------
    float
        Unbiased estimate of the maximum mean discrepancy using fewer Kernel evaluations
    
    '''

    assert n1 == n2, "n has to be equal to m"

    Kxx = K[:n1,:n1]
    Kyy = K[n1:,n1:]
    Kxy = K[:n1,n1:]
    Kyx = K[n1:,:n1]

    if n1 %2 != 0:
        n1 = n1-1
    return np.mean(Kxx[range(0,n1-1,2), range(1,n1,2)]) +\
            np.mean(Kyy[range(0,n1-1,2), range(1,n1,2)]) -\
            np.mean(Kxy[range(0,n1-1,2), range(1,n1,2)]) -\
            np.mean(Kyx[range(0,n1-1,2), range(1,n1,2)])


def factorial_k(m, k):
    """
    Calculate (m)_k := m(m-1)*...* (m-k+1)
    
    """
    
    base = m
    for i in range(1,k):
        m *= base - i
    return m


def power_ratio(K, mmd_stat, threshold, m):
    """
    The function calculates the power ratio of a specific kernel

    Parameters
    --------------------
    K: numpy array, The ernel matrix
    mmd_stat: float, the sample mmd value. Note this is the squared unbiased mmd value
    threshold: float, the test thershold for a given type I error (alpha value)
    m: int, size of sample 1

    Returns
    --------------------
    ratio: float, the power ratio, the bigger the better
    power: float, The power of the test
    


    Variance is found in Unbiased estimators for the variance of MMD estimators Danica J. Sutherland https://arxiv.org/pdf/1906.02104.pdf
    """

    Kxx = K[:m, :m]
    Kxy = K[:m, m:]
    Kyy = K[m:, m:]

    if Kxx.shape[0] != Kyy.shape[0]:
        raise ValueError("sample 1 and sample 2 should have the same size")

    H = Kxx + Kyy - Kxy - Kxy.T

    part1 = np.inner(np.sum(H,axis=1), np.sum(H,axis=1))
    part2 = np.sum(H)

    V = (4/m**3) * part1 - (4/m**4) * (part2**2)

    if V <= 0:
        raise ValueError(f"V = {V}")

    # Ktxx = Kxx.copy()
    # Ktxy = Kxy.copy()
    # Ktyy = Kyy.copy()
    # np.fill_diagonal(Ktxx, 0)
    # np.fill_diagonal(Ktxy, 0)
    # np.fill_diagonal(Ktyy, 0)



    # e = np.ones(m)
    # # Calculate variance
    # V = (
    #      (4/factorial_k(m, 4)) * (np.inner(np.matmul(Ktxx,e),np.matmul(Ktxx,e))  + np.inner(np.matmul(Ktyy,e),np.matmul(Ktyy,e)) )
    #     + ((4*(m**2 - m - 1)) / (m**3 * (m-1)**2)) * (np.inner(np.matmul(Kxy,e),np.matmul(Kxy,e))  + np.inner(np.matmul(Kxy.T,e),np.matmul(Kxy.T,e)) )
    #     - (8 / ((m**2) * (m**2 - 3*m + 2))) * (np.dot(e, Ktxx).dot(Kxy).dot(e) + np.dot(e, Ktyy).dot(Kxy.T).dot(e))
    #     + (8 / (m**2 * factorial_k(m, 3))) * ((np.dot(e, Ktxx).dot(e) + np.dot(e, Ktyy).dot(e))*np.dot(e,Kxy).dot(e))
    #     - ((2*(2*m-3))/(factorial_k(m,2)*factorial_k(m,4))) * (np.dot(e, Ktxx).dot(e)**2 + np.dot(e, Ktyy).dot(e)**2)
    #     - ((4*(2*m-3))/(m**3 * (m-1)**3)) * np.dot(e,Kxy).dot(e)**2
    #     - (2/(m *(m**3 - 6*m**2 +11*m -6))) *(np.linalg.norm(Ktxx, ord = 'fro')**2 + np.linalg.norm(Ktyy, ord = 'fro')**2)
    #     + ((4*(m-2))/(m**2 * (m-1)**3)) * np.linalg.norm(Kxy, ord='fro')**2
    # )

    

    ratio = (mmd_stat / np.sqrt(V)) - (threshold/(m*np.sqrt(V)))
    power = norm.cdf(ratio)

    return ratio, power, V





class BoostrapMethods():
    """
    Various Bootstrap/Permutation Functions

    Class for permutation testing
    """

    def __init__(self, list_of_functions:list) -> None:
        """

        Parameters
        ---------------------------
        :param list_of_functions: List of functions that should be applied to the the permutated K matrix
        boot_arg
        """

        self.list_of_functions = list_of_functions

    @staticmethod
    def issymmetric(a, rtol=1e-05, atol=1e-08):
        """
        Check if matrix is symmetric
        """
        return np.allclose(a, a.T, rtol=rtol, atol=atol)

    @staticmethod
    @njit
    def PermutationScheme(K) -> np.array:
        """
        :param K: Kernel Matrix
        """

        K_i = np.empty(K.shape)
        index = np.random.permutation(K.shape[0])
        for i in range(len(index)):
            for j in range(len(index)):
                K_i[i,j] = K[index[i], index[j]]

        return K_i

    @staticmethod
    @njit
    def BootstrapScheme(K) -> np.array:
        """
        :param K: Kernel Matrix
        """

        K_i = np.empty(K.shape)
        index = np.random.choice(K.shape[0], size = K.shape[0])
    
        for i in range(len(index)):
            for j in range(len(index)):
                K_i[i,j] = K[index[i], index[j]]

        return K_i

    @staticmethod
    def NBB(K, n, m, l):
        """
        Non over-lapping block boostrap


        Parameters
        -----------------------------
        K - n+m times n+m np array, Kernel matrix
        n - Nr in sample 1
        m - Nr in sample 2
        l - block length

        Returns
        -----------------------------
        Permutated Kernel Matrix

        """


        b_1 = int(np.ceil(n/l))
        b_2 = int(np.ceil(m/l))

        #blocks_1 = np.random.permutation(np.array(range(b_1)))
        #blocks_2 = np.random.permutation(np.array(range(b_2)))

        index_1 = np.array(range(n))
        index_2 = np.array(range(m))

        perm_1 = [index_1[(l*block):(l*block+l)] for block in range(b_1) ]
        perm_2 = [index_2[(l*block):(l*block+l )]+ n for block in range(b_2) ] 

        blocks = perm_1 + perm_2
        permutated_blocks = np.concatenate([blocks[i] for i in np.random.permutation(np.array(range(len(blocks))))])

        return K[np.ix_(permutated_blocks, permutated_blocks)]

    @staticmethod
    def MBB(K, n, m, l):
        """
        Over-lapping block boostrap


        Parameters
        -----------------------------
        K - n+m times n+m np array, Kernel matrix
        n - Nr in sample 1
        m - Nr in sample 2
        l - block length

        Returns
        -----------------------------
        Permutated Kernel Matrix

        """

        if (n <= l-1) | (m <= l-1):
            raise ValueError("Number of samples must be larger than l-1")

        # blocks_1 = np.random.permutation(np.array(range(b_1)))
        # blocks_2 = np.random.permutation(np.array(range(b_2)))

        index_1 = np.array(range(n))
        index_2 = np.array(range(m))

        perm_1 = [index_1[(i):(i+l)] for i in range(n-l+1) ]
        perm_2 = [index_2[(i):(i+l)]+ n for i in range(m-l+1) ] ## add n to get correct index of sample 2

        blocks = perm_1 + perm_2
        permutated_blocks = np.concatenate([blocks[i] for i in np.random.permutation(np.array(range(len(blocks))))])

        return K[np.ix_(permutated_blocks, permutated_blocks)], permutated_blocks



    @staticmethod
    def MMD_u_for_MBB(K, permutated_blocks, l, n, m):

        n1 = (n-l+1)*l
        m1 = (m-l+1)*l

        if (n1 + m1) != K.shape[0]:
            raise ValueError("Sizes weird")

        Kxx = K[:n1, :n1]
        Kyy = K[n1:, n1:]

        K_xx_sum = np.sum([np.sum(Kxx[i, permutated_blocks[i] != permutated_blocks[:n1]]) for i in range(n1)])
        K_yy_sum = np.sum([np.sum(Kyy[i, permutated_blocks[i + n1] != permutated_blocks[n1:]]) for i in range(m1)])

        number_x_sum = np.sum([np.sum(permutated_blocks[i] != permutated_blocks[:n1]) for i in range(n1)])
        number_y_sum = np.sum([np.sum(permutated_blocks[i + n1] != permutated_blocks[n1:]) for i in range(m1)])

        Kxy_sum = np.sum(K[:n1, n1:])
        
        return (1.0 / number_x_sum) * K_xx_sum + (1.0 / number_y_sum) * K_yy_sum - 2.0 / (n1 * m1) * Kxy_sum

    @staticmethod
    def Kernel_to_MMB(K, l, n, m):
        index_1 = np.array(range(n))
        index_2 = np.array(range(m))
        perm_1 = [index_1[(i):(i+l)] for i in range(n-l+1) ]
        perm_2 = [index_2[(i):(i+l)]+ n for i in range(m-l+1) ] ## add n to get correct index of sample 2
        idx = np.concatenate(perm_1 + perm_2)

        return K[np.ix_(idx, idx)], idx


    def Bootstrap(self, K, function_arguments,B:int, method:str = "PermutationScheme", check_symmetry:bool = False, boot_arg:dict = None) -> None:
        """

        Parameters
        --------------------
        :param K: Kernel matrix that we want to permutate
        :param function_arguments: List of dictionaries with inputs for its respective function in list_of_functions,  excluding K. If no input set as None.
        :param B: Number of Bootstraps
        :param method: Which permutation method should be applied?
        :param check_symmetry: Should the scheme check if the matrix is symmetirc, each time? Adds time complexity
        boot_arg arguments to bootstrap method
        """

        self.K = K.copy()
        self.function_arguments = function_arguments
        self.boot_arg = boot_arg
        assert self.issymmetric(self.K), "K is not symmetric"

        # keep p-value result from each MMD function
        p_value_dict = dict()
        
        # get arguments of each function ready for evaluation
        inputs = [None] * len(self.list_of_functions)
        for i in range(len(self.list_of_functions)):
            if self.function_arguments[i] is None:
                continue
            inputs[i] =  ", ".join("=".join((k,str(v))) for k,v in sorted(self.function_arguments[i].items()))

        # Calculate sample mmd statistic, and create a dictionary for bootstrapped statistics
        sample_statistic = dict()
        boot_statistic = dict()
        for i in range(len(self.list_of_functions)):
            # the key is the name of the MMD (test statistic) function
            key = self.list_of_functions[i].__name__
            if (method == 'MBB') and (key == 'MMD_u'):
                K_MMB, idx = self.Kernel_to_MMB(K, **self.boot_arg)
                sample_statistic[key] = self.MMD_u_for_MBB(K_MMB, idx, **self.boot_arg)
            elif (method == 'MBB') and (key == 'MMD_b'):
                K_MMB, idx = self.Kernel_to_MMB(K, **self.boot_arg)
                n1 = (self.boot_arg['n'] - self.boot_arg['l'] + 1)*self.boot_arg['l']
                m1 = (self.boot_arg['m'] - self.boot_arg['l'] + 1)*self.boot_arg['l']
                sample_statistic[key] = MMD_b(K_MMB, n1, m1)
            else:
                sample_statistic[key] =  self.list_of_functions[i](K, **self.function_arguments[i]) #eval(eval_string)
            
            boot_statistic[key] = np.zeros(B)


        # Get bootstrap evaluation method
        evaluation_method = getattr(self, method)

        # Now Perform Bootstraping
        for boot in range(B):
            if self.boot_arg is None:
                K_i = evaluation_method(self.K)
            elif method == 'MBB':
                K_i, index = evaluation_method(self.K, **self.boot_arg)
            else:
                K_i = evaluation_method(self.K, **self.boot_arg)
            if check_symmetry:
                if self.issymmetric(K_i):
                    warnings.warn("Not a Symmetric matrix", Warning)

            # apply each test defined in list_if_functions, and keep the bootstraped/permutated value
            for i in range(len(self.list_of_functions)):
                if (self.list_of_functions[i].__name__ == 'MMD_u') and (method == 'MBB'):
                    # NBB can not use the normal MMD_u function as it will not be unbiased due to overlaps
                    boot_statistic[self.list_of_functions[i].__name__][boot] = self.MMD_u_for_MBB(K_i, index, boot_arg['l'], boot_arg['n'], boot_arg['m'])
                elif (self.list_of_functions[i].__name__ == 'MMD_b') and (method == 'MBB'):
                    n1 = (self.boot_arg['n'] - self.boot_arg['l'] + 1)*self.boot_arg['l']
                    m1 = (self.boot_arg['m'] - self.boot_arg['l'] + 1)*self.boot_arg['l']
                    boot_statistic[self.list_of_functions[i].__name__][boot] = MMD_b(K_i, n1, m1)
                else:
                    boot_statistic[self.list_of_functions[i].__name__][boot] = self.list_of_functions[i](K_i, **self.function_arguments[i])#eval(eval_string)

        # calculate p-value
        for key in sample_statistic.keys():
            p_value_dict[key] =  (boot_statistic[key] >= sample_statistic[key]).sum()/float(B)



        self.p_values = p_value_dict
        self.sample_test_statistic = sample_statistic
        self.boot_test_statistic = boot_statistic




class DegreeGraphs():
    """
    Wrapper for a graph generator. Defines the labels and attribute generation. Used as a parent class.
    """

    def __init__(self, n, nnode, k = None, l = None,  a = None, e=None, fullyConnected = False, **kwargs) -> None:
        """
        :param kernel: Dictionary with kernel information
        :param n: Number of samples
        :param nnode: Number of nodes
        :param k: Degree
        :param l: Labelling scheme
        :param a: Attribute scheme
        :param **kwargs: Arguments for the labelling/attribute functions
        :param path: save data path
        """
        self.n = n
        self.nnode = nnode
        self.k = k
        self.l = l
        self.a = a
        self.e = e
        self.kwargs = kwargs
        self.fullyConnected = fullyConnected


    def samelabels(self, G):
        """
        labelling Scheme. All nodes get same label

        :param G: Networkx graph
        """
        return dict( ( (i, 'a') for i in range(len(G)) ) )

    def samelabels_float(self, G):
        """
        labelling Scheme. All nodes get same label

        :param G: Networkx graph
        """
        return dict( ( (i, 0.0) for i in range(len(G)) ) )

    def degreelabels(self, G):
        """
        labelling Scheme. Nodes labelled with their degree

        :param G: Networkx graph
        :return: Dictionary
        """

        nodes_degree = dict(G.degree)
        return {key: str(value) for key, value in nodes_degree.items()}

    def alldistinctlabels(self, G):
        """
        labelling Scheme. Nodes get unique labes

        :param G: Networkx graph
        :return: Dictionary
        """

        return dict( ( (i, str(i)) for i in range(len(G)) ) )

    def degreelabelsScaled(self, G):
        """
        labelling Scheme. Nodes labelled with their degree

        :param G: Networkx graph
        :return: Dictionary
        """

        nodes_degree = dict(G.degree)
        return {key: str(round(value/G.number_of_nodes(),2)) for key, value in nodes_degree.items()}

    def normattr(self, G):
        """
        labelling Scheme. Nodes labelled with their degree

        :param G: Networkx graph
        :return: Dictionary
        """
        loc = self.kwargs.get('loc', 0)
        scale = self.kwargs.get('scale', 1)
        return dict( ( (i, np.random.normal(loc = loc, scale = scale, size = (1,))) for i in range(len(G)) ) )

    def rnglabels(self, G):
        """
        labelling Scheme. Nodes labelled according to a discrete pmf

        :param G: Networkx graph
        :param pmf: pmf as list. If None then uniform over all entries
        :return: Dictionary
        """
        import string
        assert not self.kwargs['nr_letters'] is None, "Number of letters (nr_letters) has to be specified"
        

        # check if the pmf of labels has been given
        if not 'pmf' in self.kwargs.keys():
            pmf = None
        else:
            pmf = self.kwargs['pmf']
            assert  np.sum(self.kwargs['pmf']) >0.999, "pmf has to sum to 1"

        letters = list(string.ascii_lowercase[:self.kwargs['nr_letters']])
        return dict( ( (i, np.random.choice(letters, p = pmf)) for i in range(len(G)) ) )

    def normal_conditional_on_latent_mean_rv(self, G):
        """
        Generate a random variable where the mean follows another normal distribution
        scale: sd of normal
        scale_latent: sd of the latent normal
        """

        loc = np.random.normal(self.kwargs.get('loc_latent', 1), self.kwargs.get('scale_latent', 1), size = self.nnode)
        scale = self.kwargs.get('scale', 1)
        return dict( ( (i, np.random.normal(loc = loc[i], scale = scale, size = (1,))) for i in range(len(G)) ) )

    def edges(self, G):
        """
        concatenate edge labels of node and set it as the node label
        """

        return dict(( (i, ''.join(map(str,sorted([ info[2] for info in G.edges(i, data = 'sign')]))) ) for i in range(len(G))))
    
    def edge_random_sign(self,G):
        
        edge_w = dict()
        for e in G.edges():
            edge_w[e] = np.random.choice([1,-1], p=[self.kwargs['p_sign'], 1-self.kwargs['p_sign']])

        return edge_w



    def random_edge_weights(self, G):
        def edge_dist():
            from scipy.stats import uniform
            return uniform.rvs(size=1,  loc = self.kwargs['ul'] , scale = self.kwargs['uu'])[0]

        edge_w = dict()
        for e in G.edges():
            edge_w[e] = edge_dist()

        return edge_w


def scale_free(n, exponent):
    """
    Parameters:
    -------------------
    n - number of nodes
    exponent - power law exponent
    
    """
    while True:  
        s=[]
        while len(s)<n:
            nextval = int(nx.utils.powerlaw_sequence(1, exponent)[0])
            if nextval!=0:
                s.append(nextval)
        if sum(s)%2 == 0:
            break
    G = nx.configuration_model(s)
    G = nx.Graph(G) # remove parallel edges
    G.remove_edges_from(nx.selfloop_edges(G))

    return G


class ScaleFreeGraph(DegreeGraphs):
    """
    Generate a powerlaw graph
    """

    def __init__(self,  n, nnode, exponent, l = None,  a = None, e = None, e_name = 'weight', **kwargs):
        """
        Parameters:
        ---------------------
        n - number of samples
        nnode - number of nodes


        balance_target - ratio of balanced triangles to unbalanced ones.
        exponent - power law exponent

        
        """
        super().__init__( n = n, nnode = nnode, l = l ,  a = a, e=e, **kwargs )

        self.exponent = exponent
        self.e_name = e_name


    def Generate(self):

        self.Gs = []

        for _ in range(self.n):

            if self.fullyConnected:
                while True:
                    G = scale_free(self.nnode, self.exponent)
                    if nx.is_connected(G):
                        break
            else:
                G = scale_free(self.nnode, self.exponent)
            

            if not self.e is None:
                edge_weight = getattr(self, self.e)
                edge_weight_dict = edge_weight(G)
                nx.set_edge_attributes(G, values = edge_weight_dict, name = self.e_name)

            if (not self.l is None) and (not self.a is None):
                label = getattr(self, self.l)
                label_dict = label(G)
                nx.set_node_attributes(G, label_dict, 'label')
                attributes = getattr(self, self.a)
                attribute_dict = attributes(G)
                nx.set_node_attributes(G, attribute_dict, 'attr')
            elif not self.l is None:
                label = getattr(self, self.l)
                label_dict = label(G)
                nx.set_node_attributes(G, label_dict, 'label')
            elif not self.a is None:
                attributes = getattr(self, self.a)
                attribute_dict = attributes(G)
                nx.set_node_attributes(G, attribute_dict, 'attr')

            self.Gs.append(G)


class BinomialGraphs(DegreeGraphs):
    """
    Class that generates tvo samples of binomial graphs and compares them.
    """
    def __init__(self,  n, nnode, k, l = None,  a = None, **kwargs):
        super().__init__( n, nnode, k, l ,  a, **kwargs )
        self.p = k/float(nnode-1)

    def Generate(self) -> None:
        """
        :return: list of networkx graphs
        """
        self.Gs = []
        for _ in range(self.n):
            if self.fullyConnected:
                while True:
                    G = nx.fast_gnp_random_graph(self.nnode, self.p)
                    if nx.is_connected(G):
                        break
            else:
                G = nx.fast_gnp_random_graph(self.nnode, self.p)

            if (not self.l is None) and (not self.a is None):
                label = getattr(self, self.l)
                label_dict = label(G)
                nx.set_node_attributes(G, label_dict, 'label')
                attributes = getattr(self, self.a)
                attribute_dict = attributes(G)
                nx.set_node_attributes(G, attribute_dict, 'attr')
            elif not self.l is None:
                label = getattr(self, self.l)
                label_dict = label(G)
                nx.set_node_attributes(G, label_dict, 'label')
            elif not self.a is None:
                attributes = getattr(self, self.a)
                attribute_dict = attributes(G)
                nx.set_node_attributes(G, attribute_dict, 'attr')

            if not self.e is None:
                edge_weight = getattr(self, self.e)
                edge_weight_dict = edge_weight(G)
                nx.set_edge_attributes(G, values = edge_weight_dict, name = 'weight')


            self.Gs.append(G)





if __name__ == '__main__':
    nr_nodes_1 = 100
    nr_nodes_2 = 100
    n = 5
    m = 5

    average_degree = 6
    bg1 = BinomialGraphs(n, nr_nodes_1,average_degree, l = None)
