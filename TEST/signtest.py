import networkx as nx
import numpy as np


import string
import pickle # save data frame (results) in a .pkl file
import pandas as pd

import os, sys


sys.path.insert(0, 'C:/Users/User/Code/MMD_Graph_Diversification')
sys.path.insert(0, 'C:/Users/User/Code/MMD_Graph_Diversification/myKernels')
from myKernels import RandomWalk as rw
import MMDforGraphs as mg
import WWL
import sp
from multiprocessing import Pool, freeze_support
import tqdm
import grakel as gk






import collections
import warnings

import numpy as np

from itertools import chain
from collections import Counter
from numbers import Real

from sklearn.utils import check_random_state
from sklearn.preprocessing import normalize as normalizer

from grakel.graph import Graph
from grakel.kernels import Kernel

# Python 2/3 cross-compatibility import
from six import itervalues
from six import iteritems
from six.moves import filterfalse


def _dot(x, y):
    return sum(x[k]*y[k] for k in x)


class Propagation(Kernel):
    r"""The Propagation kernel for fully labeled graphs.
    See :cite:`neumann2015propagation`: Algorithms 1, 3, p. 216, 221.
    Parameters
    ----------
    t_max : int, default=5
        Maximum number of iterations.
    w : int, default=0.01
        Bin width.
    M : str, default="TV"
        The preserved distance metric (on local sensitive hashing):
            - "H": hellinger
            - "TV": total-variation
    metric : function (Counter, Counter -> number),
        default=:math:`f(x,y)=\sum_{i} x_{i}*y_{i}`
        A metric between two 1-dimensional numpy arrays of numbers that outputs a number.
        It must consider the case where the keys of y are not in x, when different features appear
        at transform.
    random_state :  RandomState or int, default=None
        A random number generator instance or an int to initialize a RandomState as a seed.
    Attributes
    ----------
    _enum_labels : dict
        Holds the enumeration of the input labels.
    _parent_labels : set
        Holds a set of the input labels.
    random_state_ : RandomState
        A RandomState object handling all randomness of the class.
    """

    _graph_format = "adjacency"
    attr_ = False

    def __init__(self,
                 n_jobs=None,
                 verbose=False,
                 normalize=False,
                 random_state=None,
                 metric=_dot,
                 M="TV",
                 t_max=5,
                 w=0.01):
        """Initialise a propagation kernel."""
        super(Propagation, self).__init__(n_jobs=n_jobs,
                                          verbose=verbose,
                                          normalize=normalize)

        self.random_state = random_state
        self.M = M
        self.t_max = t_max
        self.w = w
        self.metric = metric
        self._initialized.update({"M": False, "t_max": False, "w": False,
                                  "random_state": False, "metric": False})

    def initialize(self):
        """Initialize all transformer arguments, needing initialization."""
        super(Propagation, self).initialize()

        if not self._initialized["random_state"]:
            self.random_state_ = check_random_state(self.random_state)
            self._initialized["random_state"] = True

        if not self._initialized["metric"]:
            if (type(self.M) is not str or
                    (self.M not in ["H", "TV"] and not self.attr_) or
                    (self.M not in ["L1", "L2"] and self.attr_)):
                if self.attr_:
                    raise TypeError('Metric type must be a str, one of "L1", "L2"')
                else:
                    raise TypeError('Metric type must be a str, one of "H", "TV"')

            if not self.attr_:
                self.take_sqrt_ = self.M == "H"

            self.take_cauchy_ = self.M in ["TV", "L1"]
            self._initialized["metric"] = True

        if not self._initialized["t_max"]:
            if type(self.t_max) is not int or self.t_max <= 0:
                raise TypeError('The number of iterations must be a ' +
                                'positive integer.')
            self._initialized["t_max"] = True

        if not self._initialized["w"]:
            if not isinstance(self.w, Real) and self.w <= 0:
                raise TypeError('The bin width must be a positive number.')
            self._initialized["w"] = True

        if not self._initialized["metric"]:
            if not callable(self.metric):
                raise TypeError('The base kernel must be callable.')
            self._initialized["metric"] = True

    def pairwise_operation(self, x, y):
        """Calculate the kernel value between two elements.
        Parameters
        ----------
        x, y: list
            Inverse label dictionaries.
        Returns
        -------
        kernel : number
            The kernel value.
        """
        return sum(self.metric(x[t], y[t]) for t in range(self.t_max))

    def parse_input(self, X):
        """Parse and create features for the propation kernel.
        Parameters
        ----------
        X : iterable
            For the input to pass the test, we must have:
            Each element must be an iterable with at most three features and at
            least one. The first that is obligatory is a valid graph structure
            (adjacency matrix or edge_dictionary) while the second is
            node_labels and the third edge_labels (that correspond to the given
            graph format). A valid input also consists of graph type objects.
        Returns
        -------
        local_values : dict
            A dictionary of pairs between each input graph and a bins where the
            sampled graphlets have fallen.
        """
        if not isinstance(X, collections.Iterable):
            raise ValueError('input must be an iterable\n')
        else:
            i = -1
            transition_matrix = dict()
            labels = set()
            L = list()
            for (idx, x) in enumerate(iter(X)):
                is_iter = isinstance(x, collections.Iterable)
                if is_iter:
                    x = list(x)
                if is_iter and len(x) in [0, 2, 3, 4]:
                    if len(x) == 0:
                        warnings.warn('Ignoring empty element on ' +
                                      'index: '+str(idx))
                        continue
                    if len(x) == 2 and type(x[0]) is Graph:
                        g, T = x
                    else:
                        g = Graph(x[0], x[1], {}, self._graph_format)
                        if len(x) == 4:
                            T = x[3]
                        else:
                            T = None
                elif type(x) is Graph:
                    g, T = x, None
                else:
                    raise ValueError('Each element of X must be either a ' +
                                     'Graph or an iterable with at least 2 ' +
                                     'and at most 4 elements\n')

                if T is not None:
                    if T.shape[0] != T.shape[1]:
                        raise TypeError('Transition matrix on index' +
                                        ' ' + str(idx) + 'must be ' +
                                        'a square matrix.')
                    if T.shape[0] != g.nv():
                        raise TypeError('Propagation matrix must ' +
                                        'have the same dimension ' +
                                        'as the number of vertices.')
                else:
                    T = g.get_adjacency_matrix()

                i += 1
                transition_matrix[i] = normalizer(T, axis=1, norm='l1')
                label = g.get_labels(purpose='adjacency')
                try:
                    labels |= set(itervalues(label))
                except TypeError:
                    raise TypeError('For a non attributed kernel, labels should be hashable.')
                L.append((g.nv(), label))

            if i == -1:
                raise ValueError('Parsed input is empty')

            # The number of parsed graphs
            n = i+1

            # enumerate labels
            if self._method_calling == 1:
                enum_labels = {l: i for (i, l) in enumerate(list(labels))}
                self._enum_labels = enum_labels
                self._parent_labels = labels
            elif self._method_calling == 3:
                new_elements = labels - self._parent_labels
                if len(new_elements) > 0:
                    new_enum_labels = iter((l, i) for (i, l) in
                                           enumerate(list(new_elements), len(self._enum_labels)))
                    enum_labels = dict(chain(iteritems(self._enum_labels), new_enum_labels))
                else:
                    enum_labels = self._enum_labels

            # make a matrix for all graphs that contains label vectors
            P, data, indexes = dict(), list(), [0]
            for (k, (nv, label)) in enumerate(L):
                data += [(indexes[-1] + j, enum_labels[label[j]]) for j in range(nv)]
                indexes.append(indexes[-1] + nv)

            # Initialise the on hot vector
            rows, cols = zip(*data)
            P = np.zeros(shape=(indexes[-1], len(enum_labels)))
            P[rows, cols] = 1
            dim_orig = len(self._enum_labels)

            # feature vectors
            if self._method_calling == 1:
                # simple normal
                self._u, self._b, self._hd = list(), list(), list()
                for t in range(self.t_max):
                    u = self.random_state_.randn(len(enum_labels))

                    if self.take_cauchy_:
                        # cauchy
                        u = np.divide(u, self.random_state_.randn(len(enum_labels)))

                    self._u.append(u)
                    # random offset
                    self._b.append(self.w*self.random_state_.rand())

                phi = {k: dict() for k in range(n)}
                for t in range(self.t_max):
                    # for hash all graphs inside P and produce the feature vectors
                    hashes = self.calculate_LSH(P, self._u[t], self._b[t])
                    hd = dict((j, i) for i, j in enumerate(set(np.unique(hashes))))
                    self._hd.append(hd)
                    features = np.vectorize(lambda i: hd[i])(hashes)

                    # Accumulate the results.
                    for k in range(n):
                        phi[k][t] = Counter(features[indexes[k]:indexes[k+1]])

                    # calculate the Propagation matrix if needed
                    if t < self.t_max-1:
                        for k in range(n):
                            start, end = indexes[k:k+2]
                            P[start:end, :] = np.dot(transition_matrix[k], P[start:end, :])

                return [phi[k] for k in range(n)]

            elif (self._method_calling == 3 and dim_orig >= len(enum_labels)):
                phi = {k: dict() for k in range(n)}
                for t in range(self.t_max):
                    # for hash all graphs inside P and produce the feature vectors
                    hashes = self.calculate_LSH(P, self._u[t], self._b[t])
                    hd = dict(chain(
                            iteritems(self._hd[t]),
                            iter((j, i) for i, j in enumerate(
                                    filterfalse(lambda x: x in self._hd[t],
                                                np.unique(hashes)),
                                    len(self._hd[t])))))

                    features = np.vectorize(lambda i: hd[i])(hashes)

                    # Accumulate the results.
                    for k in range(n):
                        phi[k][t] = Counter(features[indexes[k]:indexes[k+1]])

                    # calculate the Propagation matrix if needed
                    if t < self.t_max-1:
                        for k in range(n):
                            start, end = indexes[k:k+2]
                            P[start:end, :] = np.dot(transition_matrix[k], P[start:end, :])

                return [phi[k] for k in range(n)]

            else:
                cols = np.array(cols)
                vertices = np.where(cols < dim_orig)[0]
                vertices_p = np.where(cols >= dim_orig)[0]
                nnv = len(enum_labels) - dim_orig
                phi = {k: dict() for k in range(n)}
                for t in range(self.t_max):
                    # hash all graphs inside P and produce the feature vectors
                    hashes = self.calculate_LSH(P[vertices, :dim_orig],
                                                self._u[t], self._b[t])

                    hd = dict(chain(
                            iteritems(self._hd[t]),
                            iter((j, i) for i, j in enumerate(
                                    filterfalse(lambda x: x in self._hd[t],
                                                np.unique(hashes)),
                                    len(self._hd[t])))))

                    features = np.vectorize(lambda i: hd[i], otypes=[int])(hashes)

                    # for each the new labels graph hash P and produce the feature vectors
                    u = self.random_state_.randn(nnv)
                    if self.take_cauchy_:
                        # cauchy
                        u = np.divide(u, self.random_state_.randn(nnv))

                    u = np.hstack((self._u[t], u))

                    # calculate hashes for the remaining
                    hashes = self.calculate_LSH(P[vertices_p, :], u, self._b[t])
                    hd = dict(chain(iteritems(hd), iter((j, i) for i, j in enumerate(hashes, len(hd)))))

                    features_p = np.vectorize(lambda i: hd[i], otypes=[int])(hashes)

                    # Accumulate the results
                    for k in range(n):
                        A = Counter(features[np.logical_and(
                            indexes[k] <= vertices, vertices <= indexes[k+1])])
                        B = Counter(features_p[np.logical_and(
                            indexes[k] <= vertices_p, vertices_p <= indexes[k+1])])
                        phi[k][t] = A + B

                    # calculate the Propagation matrix if needed
                    if t < self.t_max-1:
                        for k in range(n):
                            start, end = indexes[k:k+2]
                            P[start:end, :] = np.dot(transition_matrix[k], P[start:end, :])

                        Q = np.all(P[:, dim_orig:] > 0, axis=1)
                        vertices = np.where(~Q)[0]
                        vertices_p = np.where(Q)[0]

                return [phi[k] for k in range(n)]

    def calculate_LSH(self, X, u, b):
        """Calculate Local Sensitive Hashing needed for propagation kernels.
        See :cite:`neumann2015propagation`, p.12.
        Parameters
        ----------
        X : np.array
            A float array of shape (N, D) with N vertices and D features.
        u : np.array, shape=(D, 1)
            A projection vector.
        b : float
            An offset (times w).
        Returns
        -------
        lsh : np.array.
            The local sensitive hash coresponding to each vertex.
        """
        if self.take_sqrt_:
            X = np.sqrt(X)

        # hash
        return np.floor((np.dot(X, u)+b)/self.w)

class PropagationAttr(Propagation):
    r"""The Propagation kernel for fully attributed graphs.
    See :cite:`neumann2015propagation`: Algorithms 1, 3, p. 216, 221.
    Parameters
    ----------
    t_max : int, default=5
        Maximum number of iterations.
    w : int, default=0.01
        Bin width.
    M : str, default="TV"
        The preserved distance metric (on local sensitive hashing):
            - "L1": l1-norm
            - "L2": l2-norm
    metric : function (np.array, np.array -> number),
        default=:math:`f(x,y)=\sum_{i} x_{i}*y_{i}`
        A metric between two 1-dimensional numpy arrays of numbers
        that outputs a number.
    Attributes
    ----------
    M : str
        The preserved distance metric (on local sensitive hashing).
    tmax : int
        Holds the maximum number of iterations.
    w : int
        Holds the bin width.
    metric : function (np.array, np.array -> number)
        A metric between two 1-dimensional numpy arrays of numbers
        that outputs a number.
    """

    _graph_format = "adjacency"
    attr_ = True

    def __init__(self,
                 n_jobs=None,
                 verbose=False,
                 normalize=False,
                 random_state=None,
                 metric=_dot,
                 M="L1",
                 t_max=5,
                 w=4):
        """Initialise a propagation kernel."""
        super(PropagationAttr, self).__init__(n_jobs=n_jobs,
                                              verbose=verbose,
                                              normalize=normalize,
                                              random_state=random_state,
                                              metric=metric,
                                              M=M,
                                              t_max=t_max,
                                              w=w)

    def initialize(self):
        """Initialize all transformer arguments, needing initialization."""
        super(PropagationAttr, self).initialize()

    def parse_input(self, X):
        """Parse and create features for the attributed propation kernel.
        Parameters
        ----------
        X : iterable
            For the input to pass the test, we must have:
            Each element must be an iterable with at most three features and at
            least one. The first that is obligatory is a valid graph structure
            (adjacency matrix or edge_dictionary) while the second is
            node_labels and the third edge_labels (that correspond to the given
            graph format). A valid input also consists of graph type objects.
        Returns
        -------
        local_values : dict
            A dictionary of pairs between each input graph and a bins where the
            sampled graphlets have fallen.
        """
        if not isinstance(X, collections.Iterable):
            raise ValueError('input must be an iterable\n')
        else:
            # The number of parsed graphs
            n = 0
            transition_matrix = dict()
            indexes = [0]
            Attr = list()
            for (idx, x) in enumerate(iter(X)):
                is_iter = isinstance(x, collections.Iterable)
                if is_iter:
                    x = list(x)
                if is_iter and len(x) in [0, 2, 3, 4]:
                    if len(x) == 0:
                        warnings.warn('Ignoring empty element on ' +
                                      'index: '+str(idx))
                        continue
                    if len(x) == 2 and type(x[0]) is Graph:
                        g, T = x
                    else:
                        g = Graph(x[0], x[1], {}, self._graph_format)
                        if len(x) == 4:
                            T = x[3]
                        else:
                            T = None
                elif type(x) is Graph:
                    g, T = x, None
                else:
                    raise ValueError('Each element of X must be either a ' +
                                     'Graph or an iterable with at least 2 ' +
                                     'and at most 4 elements\n')

                if T is not None:
                    if T.shape[0] != T.shape[1]:
                        raise TypeError('Transition matrix on index' +
                                        ' ' + str(idx) + 'must be ' +
                                        'a square matrix.')
                    if T.shape[0] != g.nv():
                        raise TypeError('Propagation matrix must ' +
                                        'have the same dimension ' +
                                        'as the number of vertices.')
                else:
                    T = g.get_adjacency_matrix()

                nv = g.nv()
                transition_matrix[n] = normalizer(T, axis=1, norm='l1')
                attr = g.get_labels(purpose="adjacency")
                try:
                    attributes = np.array([attr[j] for j in range(nv)])
                except TypeError:
                    raise TypeError('All attributes of a single graph should have the same dimension.')

                Attr.append(attributes)
                indexes.append(indexes[-1] + nv)
                n += 1
            try:
                P = np.vstack(Attr)
            except ValueError:
                raise ValueError('Attribute dimensions should be the same, for all graphs')

            if self._method_calling == 1:
                self._dim = P.shape[1]
            else:
                if self._dim != P.shape[1]:
                    raise ValueError('transform attribute vectors should'
                                     'have the same dimension as in fit')

            if n == 0:
                raise ValueError('Parsed input is empty')

            # feature vectors
            if self._method_calling == 1:
                # simple normal
                self._u, self._b, self._hd = list(), list(), list()
                for t in range(self.t_max):
                    u = self.random_state_.randn(self._dim)
                    if self.take_cauchy_:
                        # cauchy
                        u = np.divide(u, self.random_state_.randn(self._dim))

                    self._u.append(u)
                    # random offset
                    self._b.append(self.w*self.random_state_.randn(self._dim))

                phi = {k: dict() for k in range(n)}
                for t in range(self.t_max):
                    # for hash all graphs inside P and produce the feature vectors
                    hashes = self.calculate_LSH(P, self._u[t], self._b[t]).tolist()

                    hd = {j: i for i, j in enumerate({tuple(l) for l in hashes})}
                    self._hd.append(hd)

                    features = np.array([hd[tuple(l)] for l in hashes])

                    # Accumulate the results.
                    for k in range(n):
                        phi[k][t] = Counter(features[indexes[k]:indexes[k+1]].flat)

                    # calculate the Propagation matrix if needed
                    if t < self.t_max-1:
                        for k in range(n):
                            start, end = indexes[k:k+2]
                            P[start:end, :] = np.dot(transition_matrix[k], P[start:end, :])

                return [phi[k] for k in range(n)]

            if self._method_calling == 3:
                phi = {k: dict() for k in range(n)}
                for t in range(self.t_max):
                    # for hash all graphs inside P and produce the feature vectors
                    hashes = self.calculate_LSH(P, self._u[t], self._b[t]).tolist()

                    hd = dict(chain(
                            iteritems(self._hd[t]),
                            iter((j, i) for i, j in enumerate(
                                    filterfalse(lambda x: x in self._hd[t],
                                                {tuple(l) for l in hashes}),
                                    len(self._hd[t])))))

                    features = np.array([hd[tuple(l)] for l in hashes])

                    # Accumulate the results.
                    for k in range(n):
                        phi[k][t] = Counter(features[indexes[k]:indexes[k+1]])

                    # calculate the Propagation matrix if needed
                    if t < self.t_max-1:
                        for k in range(n):
                            start, end = indexes[k:k+2]
                            P[start:end, :] = np.dot(transition_matrix[k], P[start:end, :])

                return [phi[k] for k in range(n)]

    def calculate_LSH(self, X, u, b):
        """Calculate Local Sensitive Hashing needed for propagation kernels.
        See :cite:`neumann2015propagation`, p.12.
        Parameters
        ----------
        X : np.array
            A float array of shape (N, D) with N vertices and D features.
        u : np.array, shape=(D, 1)
            A projection vector.
        b : float
            An offset (times w).
        Returns
        -------
        lsh : np.array.
            The local sensitive hash coresponding to each vertex.
        """
        return np.floor((X*u+b)/self.w)


def iteration_weight_sign(N, B, bg1, bg2, scale2, p2):


    prop_w = [0.1, 0.01, 0.001, 0.0001]
    pyramid_L = [2,6,10]
    wl_itr = [2,4]

    out_p_val = {'rw_weight':[],
                'rw_signed':[],
                'sp':[],
                'rw_binary':[],
                'rw_abs':[],
                'Bonferroni':[],
                'Tensor':[]}

    for w in prop_w:
        out_p_val['propagation' + str(w)] = []
        out_p_val['wwl' + str(w)] = []


    for L in pyramid_L:
        out_p_val['pyramid' + str(L)+'wlab'] = []
        out_p_val['pyramid' + str(L)] = []

    for h in wl_itr:
        out_p_val['wl' + str(h)] = []
        out_p_val['wloa' + str(h)] = []


    for _ in tqdm.tqdm(range(N)):
        bg1.Generate()
        bg2.Generate()
        # Add weights
        def edge_dist( scale ):
            from scipy.stats import uniform
            return np.random.exponential(scale = scale)# uniform.rvs(size=1,  loc = loc , scale = scale)[0]
        def add_weight(G, scale ):
            edge_w = dict()
            for e in G.edges():
                edge_w[e] = edge_dist(scale)
            return edge_w

        G1 = bg1.Gs.copy()
        G2 = bg2.Gs.copy()



        for G in G1:
            nx.set_edge_attributes(G, add_weight(G, scale = 3000), "weight")
        for G in G2:
            nx.set_edge_attributes(G, add_weight(G, scale = scale2), "weight")

        for G in G1:
            nx.set_node_attributes(G, {i:str(k) for i,k in G.degree}, "label")
        for G in G2:
            nx.set_node_attributes(G, {i:str(k) for i,k in G.degree}, "label")

        
        for G in G1:
            for e in G.edges():
                if np.random.uniform() <0.35:
                    w = G.edges[e[0], e[1]]['weight']
                    G.edges[e[0], e[1]]['weight'] = -w 

        for G in G2:
            for e in G.edges():
                if np.random.uniform() < p2:
                    w = G.edges[e[0], e[1]]['weight']
                    G.edges[e[0], e[1]]['weight'] = -w 
                    
        for G in G1:
            nx.set_edge_attributes(G, {(n1, n2): np.sign(w) for n1, n2, w in G.edges().data('weight')}, "sign")
        for G in G2:
            nx.set_edge_attributes(G, {(n1, n2): np.sign(w) for n1, n2, w in G.edges().data('weight')}, "sign")

        Gs = G1 + G2
        
        Gs_abs = [nx.from_numpy_array(np.abs(nx.adjacency_matrix(G))) for G in Gs]
        Gs_binary = [nx.from_numpy_array(np.sign(nx.adjacency_matrix(G).todense())) for G in Gs]
        Gs_plus = []
        Gs_negative = []
        for i in range(len(Gs)):
            A = nx.adjacency_matrix(Gs[i]).todense()
            A_plus = A.copy()
            A_plus[A_plus<0] =0
            Gs_plus.append(nx.from_numpy_array(A_plus))
            nx.set_node_attributes(Gs_plus[i], {j:str(k) for j,k in Gs[i].degree}, "label")


            A_negative = A.copy()
            A_negative[A_negative>0] =0
            A_negative = np.abs(A_negative)
            Gs_negative.append(nx.from_numpy_array(A_negative))
            nx.set_node_attributes(Gs_negative[i], {j:str(k) for j,k in Gs[i].degree}, "label")


        #print("Graph preperation done...")   
        MMD_functions = [mg.MMD_b, mg.MMD_u, mg.MMD_l]#, mg.MONK_EST]
        kernel_hypothesis = mg.BoostrapMethods(MMD_functions)
        function_arguments = [dict(n = bg1.n, m = bg1.n  ), 
                    dict(n = bg1.n, m = bg1.n ),
                    dict(n = bg1.n, m = bg1.n )]#, 
                    #dict(Q = 3, y1 = Gs[:bg1.n], y2 = Gs[bg1.n:] )]


        #print("RW as-is")
        rw_kernel = rw.RandomWalk(Gs, c = 1e-10, normalize=0)
        K_rw = rw_kernel.fit_ARKU_plus(r=6, verbose = False, edge_attr='weight')
        v,_ = np.linalg.eigh(K_rw)
        # v[np.abs(v) < 10e-5] = 0
        if np.any(v < -10e-8):
            raise ValueError("Not psd WEIGHT")
        kernel_hypothesis.Bootstrap(K_rw, function_arguments, B = B)
        out_p_val['rw_weight'].append(kernel_hypothesis.p_values["MMD_u"])
        # print(f'p_value {kernel_hypothesis.p_values["MMD_u"]}')


        #print("Signed RW")
        rw_kernel = rw.RandomWalk(Gs, c = 1e-10, normalize=0)
        K_sign = rw_kernel.fit_ARKU_edge(r=6, verbose = False, edge_attr='weight', edge_labels=[1.0,-1.0], edge_label_tag = 'sign')
        v,_ = np.linalg.eigh(K_sign)
        # v[np.abs(v) < 10e-5] = 0
        if np.any(v < -10e-8):
            raise ValueError("Not psd SIGN EDGE")
        kernel_hypothesis.Bootstrap(K_sign, function_arguments, B = B)
        out_p_val['rw_signed'].append(kernel_hypothesis.p_values["MMD_u"])
        # print(f'p_value {kernel_hypothesis.p_values["MMD_u"]}')

        #print("RW absolute value")
        rw_kernel = rw.RandomWalk(Gs_abs, c = 1e-10, normalize=0)
        K = rw_kernel.fit_ARKU_plus(r=6, verbose = False, edge_attr='weight')
        v,_ = np.linalg.eigh(K)
        # v[np.abs(v) < 10e-5] = 0
        if np.any(v < -10e-8):
            raise ValueError("Not psd ARKU")
        kernel_hypothesis.Bootstrap(K, function_arguments, B = B)
        out_p_val['rw_abs'].append(kernel_hypothesis.p_values["MMD_u"])
        # print(f'p_value {kernel_hypothesis.p_values["MMD_u"]}')

        rw_kernel = rw.RandomWalk(Gs_binary, c = 1e-10, normalize=0)
        K = rw_kernel.fit_ARKU_plus(r=6, verbose = False, edge_attr=None)
        v,_ = np.linalg.eigh(K)
        # v[np.abs(v) < 10e-5] = 0
        if np.any(v < -10e-8):
            raise ValueError("Not psd BINARY")
        kernel_hypothesis.Bootstrap(K, function_arguments, B = B)
        out_p_val['rw_binary'].append(kernel_hypothesis.p_values["MMD_u"])
        # print(f'p_value {kernel_hypothesis.p_values["MMD_u"]}')

        
        #print("Bonferroni")
        rw_kernel = rw.RandomWalk(Gs_plus, c = 1e-10, normalize=0)
        K1 = rw_kernel.fit_ARKU_plus(r=6, verbose = False, edge_attr='weight')
        v,_ = np.linalg.eigh(K1)
        # v[np.abs(v) < 10e-5] = 0
        if np.any(v < -10e-8):
            raise ValueError("Not psd BON 1")
        kernel_hypothesis.Bootstrap(K1, function_arguments, B = B)
        pval_1 = kernel_hypothesis.p_values["MMD_u"]
        rw_kernel = rw.RandomWalk(Gs_negative, c = 1e-10, normalize=0)
        K2 = rw_kernel.fit_ARKU_plus(r=6, verbose = False, edge_attr='weight')
        v,_ = np.linalg.eigh(K2)
        # v[np.abs(v) < 10e-5] = 0
        if np.any(v < -10e-8):
            raise ValueError("Not psd BON 2")
        kernel_hypothesis.Bootstrap(K2, function_arguments, B = B)
        pval_2 = kernel_hypothesis.p_values["MMD_u"]
        out_p_val['Bonferroni'].append(np.max((pval_1,pval_2)))
        #print(f'p_value {pval_1} {pval_2}')

        #print("Tensor")
        K = np.multiply(K1,K2)
        v,_ = np.linalg.eigh(K)
        # v[np.abs(v) < 10e-5] = 0
        if np.any(v < -10e-8):
            raise ValueError("Not psd TENSOR")
        kernel_hypothesis.Bootstrap(K, function_arguments, B = B)
        # print(f'p_value {kernel_hypothesis.p_values["MMD_u"]}')
        out_p_val['Tensor'].append(kernel_hypothesis.p_values["MMD_u"])

        for L in pyramid_L:
            pm = gk.PyramidMatch(with_labels = False, L = L)
            gk_gs = gk.graph_from_networkx(Gs_plus, edge_weight_tag='weight',  node_labels_tag = 'label')
            K_plus = pm.fit_transform(gk_gs)

            pm = gk.PyramidMatch(with_labels = False, L = L)
            gk_gs = gk.graph_from_networkx(Gs_negative, edge_weight_tag='weight',  node_labels_tag = 'label')
            K_negative = pm.fit_transform(gk_gs)

            K = np.multiply(K_plus, K_negative)
            v,_ = np.linalg.eigh(K)
            if np.any(v < -10e-8):
                raise ValueError("Not psd pyramid")

            kernel_hypothesis.Bootstrap(K, function_arguments, B = B)
            out_p_val['pyramid' + str(L)].append(kernel_hypothesis.p_values["MMD_u"])   

        for L in pyramid_L:
            pm = gk.PyramidMatch(with_labels = True, L = L)
            gk_gs = gk.graph_from_networkx(Gs_plus, edge_weight_tag='weight',  node_labels_tag = 'label')
            K_plus = pm.fit_transform(gk_gs)

            pm = gk.PyramidMatch(with_labels = True, L = L)
            gk_gs = gk.graph_from_networkx(Gs_negative, edge_weight_tag='weight',  node_labels_tag = 'label')
            K_negative = pm.fit_transform(gk_gs)

            K = np.multiply(K_plus, K_negative)
            v,_ = np.linalg.eigh(K)
            if np.any(v < -10e-8):
                raise ValueError("Not psd pyramid LABEL")

            kernel_hypothesis.Bootstrap(K, function_arguments, B = B)
            out_p_val['pyramid' + str(L)+'wlab'].append(kernel_hypothesis.p_values["MMD_u"])


        for w in prop_w:
            prop = Propagation(w = w,t_max = 4)
            gk_gs = gk.graph_from_networkx(Gs_plus, edge_weight_tag='weight',  node_labels_tag = 'label')
            K_plus = prop.fit_transform(gk_gs)

            prop = Propagation(w = w,t_max = 4)
            gk_gs = gk.graph_from_networkx(Gs_negative, edge_weight_tag='weight',  node_labels_tag = 'label')
            K_negative = prop.fit_transform(gk_gs)

            K = np.multiply(K_plus, K_negative)
            v,_ = np.linalg.eigh(K)
            if np.any(v < -10e-8):
                raise ValueError("Not psd prop")

            kernel_hypothesis.Bootstrap(K, function_arguments, B = B)
            out_p_val['propagation' + str(w)].append(kernel_hypothesis.p_values["MMD_u"])

        def edge_kernel(x,y):
            return np.exp(-0.01*np.abs(x-y))
        my_sp = sp.sp_kernel(weight = 'weight', edge_kernel= edge_kernel)
        K_plus = my_sp.fit_transform(Gs_plus,verbose = False)
        K_negative = my_sp.fit_transform(Gs_negative,verbose = False)
        K = np.multiply(K_plus, K_negative)
        v,_ = np.linalg.eigh(K)
        if np.any(v < -10e-8):
            raise ValueError("Not psd sp")

        kernel_hypothesis.Bootstrap(K, function_arguments, B = B)
        out_p_val['sp'].append(kernel_hypothesis.p_values["MMD_u"])




        for h in wl_itr:
            kernel = [{"name": "weisfeiler_lehman", "n_iter":h}, {"name": "vertex_histogram"}]
            init_kernel = gk.GraphKernel(kernel= kernel, normalize=0)
            graph_list = gk.graph_from_networkx(Gs_plus, node_labels_tag='label')  # Convert to graphs to Grakel format
            K_plus = init_kernel.fit_transform(graph_list)

            kernel = [{"name": "weisfeiler_lehman", "n_iter":h}, {"name": "vertex_histogram"}]
            init_kernel = gk.GraphKernel(kernel= kernel, normalize=0)
            graph_list = gk.graph_from_networkx(Gs_negative, node_labels_tag='label')  # Convert to graphs to Grakel format
            K_negative = init_kernel.fit_transform(graph_list)

            K = np.multiply(K_plus, K_negative)
            v,_ = np.linalg.eigh(K)
            if np.any(v < -10e-8):
                raise ValueError("Not psd wl")

            kernel_hypothesis.Bootstrap(K, function_arguments, B = B)
            out_p_val['wl'+str(h)].append(kernel_hypothesis.p_values["MMD_u"])

        for h in wl_itr:
            kernel = [{"name": "WL-OA", "n_iter": h}]
            init_kernel = gk.GraphKernel(kernel= kernel, normalize=0)
            graph_list = gk.graph_from_networkx(Gs_plus, node_labels_tag='label')  # Convert to graphs to Grakel format
            K_plus = init_kernel.fit_transform(graph_list)

            kernel = [{"name": "WL-OA", "n_iter": h}]
            init_kernel = gk.GraphKernel(kernel= kernel, normalize=0)
            graph_list = gk.graph_from_networkx(Gs_negative, node_labels_tag='label')  # Convert to graphs to Grakel format
            K_negative = init_kernel.fit_transform(graph_list)

            K = np.multiply(K_plus, K_negative)
            v,_ = np.linalg.eigh(K)
            if np.any(v < -10e-8):
                raise ValueError("Not psd wloa")
            kernel_hypothesis.Bootstrap(K, function_arguments, B = B)
            out_p_val['wloa'+str(h)].append(kernel_hypothesis.p_values["MMD_u"])

        for w in prop_w:
            kernel = WWL.WWL(param = {'discount':w,'h':2, 'sinkhorn':False })
            K_plus = kernel.fit_transform(Gs_plus)

            kernel = WWL.WWL(param = {'discount':w,'h':2, 'sinkhorn':False })
            K_negative = kernel.fit_transform(Gs_negative)
            v,_ = np.linalg.eigh(K)
            if np.any(v < -10e-8):
                raise ValueError("Not psd")
            kernel_hypothesis.Bootstrap(K, function_arguments, B = B)
            out_p_val['wwl'+str(w)].append(kernel_hypothesis.p_values["MMD_u"])




    return pd.DataFrame(out_p_val)


def iteration_weight_sign_attr(N, B, bg1, bg2, scale2, loc_attr, scale_attr, p2):

    def edge_dist( scale ):
        from scipy.stats import uniform
        return np.random.exponential(scale = scale)# uniform.rvs(size=1,  loc = loc , scale = scale)[0]
    def add_weight(G, scale ):
        edge_w = dict()
        for e in G.edges():
            edge_w[e] = edge_dist(scale)
        return edge_w


    prop_w = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    pyramid_L = [2,6,10]
    wl_itr = [2,4]

    out_p_val = {'rw_weight':[],
                'rw_signed':[],
                #'sp':[],
                'rw_binary':[],
                'rw_abs':[],
                'Bonferroni':[],
                'Tensor':[]}

    for w in prop_w:
        out_p_val['propagation' + str(w)] = []
        out_p_val['wwl' + str(w)] = []


    for L in pyramid_L:
        out_p_val['pyramid' + str(L)+'wlab'] = []
        out_p_val['pyramid' + str(L)] = []

    for h in wl_itr:
        out_p_val['wl' + str(h)] = []
        out_p_val['wloa' + str(h)] = []



    for _ in tqdm.tqdm(range(N)):
        bg1.Generate()
        bg2.Generate()

        G1 = bg1.Gs.copy()
        G2 = bg2.Gs.copy()

        for G in G1:
            nx.set_edge_attributes(G, add_weight(G, scale = 3000), "weight")
        for G in G2:
            nx.set_edge_attributes(G, add_weight(G, scale = scale2), "weight")

        for G in G1:
            nx.set_node_attributes(G, {i:str(k) for i,k in G.degree}, "label")
        for G in G2:
            nx.set_node_attributes(G, {i:str(k) for i,k in G.degree}, "label")

        
        for G in G1:
            for e in G.edges():
                if np.random.uniform() <0.35:
                    w = G.edges[e[0], e[1]]['weight']
                    G.edges[e[0], e[1]]['weight'] = -w 

        for G in G2:
            for e in G.edges():
                if np.random.uniform() < p2:
                    w = G.edges[e[0], e[1]]['weight']
                    G.edges[e[0], e[1]]['weight'] = -w 

        for G in G1:
            nx.set_node_attributes(G, {i:np.random.normal(size = (1,), loc = 0.00038, scale= 0.01) for i in range(G.number_of_nodes())}, "attr")
        for G in G2:
            nx.set_node_attributes(G, {i:np.random.normal(size = (1,), loc = loc_attr, scale= scale_attr) for i in range(G.number_of_nodes())}, "attr")
                    
        for G in G1:
            nx.set_edge_attributes(G, {(n1, n2): np.sign(w) for n1, n2, w in G.edges().data('weight')}, "sign")
        for G in G2:
            nx.set_edge_attributes(G, {(n1, n2): np.sign(w) for n1, n2, w in G.edges().data('weight')}, "sign")

        Gs = G1 + G2
        
        Gs_abs = [nx.from_numpy_array(np.abs(nx.adjacency_matrix(G))) for G in Gs]
        Gs_binary = [nx.from_numpy_array(np.sign(nx.adjacency_matrix(G).todense())) for G in Gs]
        Gs_plus = []
        Gs_negative = []
        for i in range(len(Gs)):
            A = nx.adjacency_matrix(Gs[i]).todense()
            A_plus = A.copy()
            A_plus[A_plus<0] =0
            Gs_plus.append(nx.from_numpy_array(A_plus))
            nx.set_node_attributes(Gs_plus[i], nx.get_node_attributes(Gs[i], 'attr'), "attr")
            nx.set_node_attributes(Gs_plus[i], {j:str(k) for j,k in Gs[i].degree}, "label")


            A_negative = A.copy()
            A_negative[A_negative>0] =0
            A_negative = np.abs(A_negative)
            Gs_negative.append(nx.from_numpy_array(A_negative))
            nx.set_node_attributes(Gs_negative[i], nx.get_node_attributes(Gs[i], 'attr'), "attr")
            nx.set_node_attributes(Gs_negative[i], {j:str(k) for j,k in Gs[i].degree}, "label")


        #print("Graph preperation done...")   
        MMD_functions = [mg.MMD_b, mg.MMD_u, mg.MMD_l]#, mg.MONK_EST]
        kernel_hypothesis = mg.BoostrapMethods(MMD_functions)
        function_arguments = [dict(n = bg1.n, m = bg1.n  ), 
                    dict(n = bg1.n, m = bg1.n ),
                    dict(n = bg1.n, m = bg1.n )]#, 
                    #dict(Q = 3, y1 = Gs[:bg1.n], y2 = Gs[bg1.n:] )]

        p = [np.resize(list(nx.get_node_attributes(G, 'attr').values()), new_shape = (20)) for G in Gs]
        try:
        #print("Signed RW")
            rw_kernel = rw.RandomWalk(Gs, c = 1e-10, normalize=0, p = p, q = p)
            K_sign = rw_kernel.fit_ARKU_edge(r=6, verbose = False, edge_attr='weight', edge_labels=[1.0,-1.0], edge_label_tag = 'sign')
            v,_ = np.linalg.eigh(K_sign)
            # v[np.abs(v) < 10e-5] = 0
            if np.any(v < -10e-8):
                raise ValueError("Not psd SIGN EDGE")
            kernel_hypothesis.Bootstrap(K_sign, function_arguments, B = B)
            out_p_val['rw_signed'].append(kernel_hypothesis.p_values["MMD_u"])
        except:
            continue


        #print("RW as-is")
        rw_kernel = rw.RandomWalk(Gs, c = 1e-10, normalize=0, p = p, q = p)
        K_rw = rw_kernel.fit_ARKU_plus(r=6, verbose = False, edge_attr='weight')
        v,_ = np.linalg.eigh(K_rw)
        # v[np.abs(v) < 10e-5] = 0
        if np.any(v < -10e-8):
            raise ValueError("Not psd WEIGHT")
        kernel_hypothesis.Bootstrap(K_rw, function_arguments, B = B)
        out_p_val['rw_weight'].append(kernel_hypothesis.p_values["MMD_u"])
        # print(f'p_value {kernel_hypothesis.p_values["MMD_u"]}')



        #print("RW absolute value")
        rw_kernel = rw.RandomWalk(Gs_abs, c = 1e-10, normalize=0, p = p, q = p)
        K = rw_kernel.fit_ARKU_plus(r=6, verbose = False, edge_attr='weight')
        v,_ = np.linalg.eigh(K)
        # v[np.abs(v) < 10e-5] = 0
        if np.any(v < -10e-8):
            raise ValueError("Not psd ARKU")
        kernel_hypothesis.Bootstrap(K, function_arguments, B = B)
        out_p_val['rw_abs'].append(kernel_hypothesis.p_values["MMD_u"])
        # print(f'p_value {kernel_hypothesis.p_values["MMD_u"]}')

        rw_kernel = rw.RandomWalk(Gs_binary, c = 1e-10, normalize=0, p = p, q = p)
        K = rw_kernel.fit_ARKU_plus(r=6, verbose = False, edge_attr=None)
        v,_ = np.linalg.eigh(K)
        # v[np.abs(v) < 10e-5] = 0
        if np.any(v < -10e-8):
            raise ValueError("Not psd BINARY")
        kernel_hypothesis.Bootstrap(K, function_arguments, B = B)
        out_p_val['rw_binary'].append(kernel_hypothesis.p_values["MMD_u"])
        # print(f'p_value {kernel_hypothesis.p_values["MMD_u"]}')

        
        #print("Bonferroni")
        rw_kernel = rw.RandomWalk(Gs_plus, c = 1e-10, normalize=0, p = p, q = p)
        K1 = rw_kernel.fit_ARKU_plus(r=6, verbose = False, edge_attr='weight')
        v,_ = np.linalg.eigh(K1)
        # v[np.abs(v) < 10e-5] = 0
        if np.any(v < -10e-8):
            raise ValueError("Not psd BON 1")
        kernel_hypothesis.Bootstrap(K1, function_arguments, B = B)
        pval_1 = kernel_hypothesis.p_values["MMD_u"]
        rw_kernel = rw.RandomWalk(Gs_negative, c = 1e-10, normalize=0, p = p, q = p)
        K2 = rw_kernel.fit_ARKU_plus(r=6, verbose = False, edge_attr='weight')
        v,_ = np.linalg.eigh(K2)
        # v[np.abs(v) < 10e-5] = 0
        if np.any(v < -10e-8):
            raise ValueError("Not psd BON 2")
        kernel_hypothesis.Bootstrap(K2, function_arguments, B = B)
        pval_2 = kernel_hypothesis.p_values["MMD_u"]
        out_p_val['Bonferroni'].append(np.max((pval_1,pval_2)))
        #print(f'p_value {pval_1} {pval_2}')

        #print("Tensor")
        K = np.multiply(K1,K2)
        v,_ = np.linalg.eigh(K)
        # v[np.abs(v) < 10e-5] = 0
        if np.any(v < -10e-8):
            raise ValueError("Not psd TENSOR")
        kernel_hypothesis.Bootstrap(K, function_arguments, B = B)
        # print(f'p_value {kernel_hypothesis.p_values["MMD_u"]}')
        out_p_val['Tensor'].append(kernel_hypothesis.p_values["MMD_u"])

        for L in pyramid_L:
            pm = gk.PyramidMatch(with_labels = False, L = L)
            gk_gs = gk.graph_from_networkx(Gs_plus, edge_weight_tag='weight',  node_labels_tag = 'label')
            K_plus = pm.fit_transform(gk_gs)

            pm = gk.PyramidMatch(with_labels = False, L = L)
            gk_gs = gk.graph_from_networkx(Gs_negative, edge_weight_tag='weight',  node_labels_tag = 'label')
            K_negative = pm.fit_transform(gk_gs)

            K = np.multiply(K_plus, K_negative)
            v,_ = np.linalg.eigh(K)
            if np.any(v < -10e-8):
                raise ValueError("Not psd pyramid")

            kernel_hypothesis.Bootstrap(K, function_arguments, B = B)
            out_p_val['pyramid' + str(L)].append(kernel_hypothesis.p_values["MMD_u"])   

        for L in pyramid_L:
            pm = gk.PyramidMatch(with_labels = True, L = L)
            gk_gs = gk.graph_from_networkx(Gs_plus, edge_weight_tag='weight',  node_labels_tag = 'label')
            K_plus = pm.fit_transform(gk_gs)

            pm = gk.PyramidMatch(with_labels = True, L = L)
            gk_gs = gk.graph_from_networkx(Gs_negative, edge_weight_tag='weight',  node_labels_tag = 'label')
            K_negative = pm.fit_transform(gk_gs)

            K = np.multiply(K_plus, K_negative)
            v,_ = np.linalg.eigh(K)
            if np.any(v < -10e-8):
                raise ValueError("Not psd pyramid LABEL")

            kernel_hypothesis.Bootstrap(K, function_arguments, B = B)
            out_p_val['pyramid' + str(L)+'wlab'].append(kernel_hypothesis.p_values["MMD_u"])


        for w in prop_w:
            prop = PropagationAttr(w = w,t_max = 4)
            gk_gs = gk.graph_from_networkx(Gs_plus, edge_weight_tag='weight',  node_labels_tag = 'attr')
            K_plus = prop.fit_transform(gk_gs)

            prop = PropagationAttr(w = w,t_max = 4)
            gk_gs = gk.graph_from_networkx(Gs_negative, edge_weight_tag='weight',  node_labels_tag = 'attr')
            K_negative = prop.fit_transform(gk_gs)

            K = np.multiply(K_plus, K_negative)
            v,_ = np.linalg.eigh(K)
            if np.any(v < -10e-8):
                raise ValueError("Not psd prop")

            kernel_hypothesis.Bootstrap(K, function_arguments, B = B)
            out_p_val['propagation' + str(w)].append(kernel_hypothesis.p_values["MMD_u"])

        
        # def edge_kernel(x,y):
        #     return np.exp(-0.01*np.abs(x-y))
        # def node_kernel(x,y):
        #     return np.exp(-1000*np.abs(x-y))
        # my_sp = sp.sp_kernel(weight = 'weight', with_labels = True,  edge_kernel = edge_kernel, node_kernel = node_kernel)
        # K_plus = my_sp.fit_transform(Gs_plus,verbose = False)
        # K_negative = my_sp.fit_transform(Gs_negative,verbose = False)
        # K = np.multiply(K_plus, K_negative)
        # v,_ = np.linalg.eigh(K)
        # if np.any(v < -10e-8):
        #     raise ValueError("Not psd sp")

        # kernel_hypothesis.Bootstrap(K, function_arguments, B = B)
        # out_p_val['sp'].append(kernel_hypothesis.p_values["MMD_u"])




        for h in wl_itr:
            kernel = [{"name": "weisfeiler_lehman", "n_iter":h}, {"name": "vertex_histogram"}]
            init_kernel = gk.GraphKernel(kernel= kernel, normalize=0)
            graph_list = gk.graph_from_networkx(Gs_plus, node_labels_tag='label')  # Convert to graphs to Grakel format
            K_plus = init_kernel.fit_transform(graph_list)

            kernel = [{"name": "weisfeiler_lehman", "n_iter":h}, {"name": "vertex_histogram"}]
            init_kernel = gk.GraphKernel(kernel= kernel, normalize=0)
            graph_list = gk.graph_from_networkx(Gs_negative, node_labels_tag='label')  # Convert to graphs to Grakel format
            K_negative = init_kernel.fit_transform(graph_list)

            K = np.multiply(K_plus, K_negative)
            v,_ = np.linalg.eigh(K)
            if np.any(v < -10e-8):
                raise ValueError("Not psd wl")

            kernel_hypothesis.Bootstrap(K, function_arguments, B = B)
            out_p_val['wl'+str(h)].append(kernel_hypothesis.p_values["MMD_u"])

        for h in wl_itr:
            kernel = [{"name": "WL-OA", "n_iter": h}]
            init_kernel = gk.GraphKernel(kernel= kernel, normalize=0)
            graph_list = gk.graph_from_networkx(Gs_plus, node_labels_tag='label')  # Convert to graphs to Grakel format
            K_plus = init_kernel.fit_transform(graph_list)

            kernel = [{"name": "WL-OA", "n_iter": h}]
            init_kernel = gk.GraphKernel(kernel= kernel, normalize=0)
            graph_list = gk.graph_from_networkx(Gs_negative, node_labels_tag='label')  # Convert to graphs to Grakel format
            K_negative = init_kernel.fit_transform(graph_list)

            K = np.multiply(K_plus, K_negative)
            v,_ = np.linalg.eigh(K)
            if np.any(v < -10e-8):
                raise ValueError("Not psd wloa")
            kernel_hypothesis.Bootstrap(K, function_arguments, B = B)
            out_p_val['wloa'+str(h)].append(kernel_hypothesis.p_values["MMD_u"])

        for w in prop_w:
            kernel = WWL.WWL(param = {'discount':w,'h':2, 'sinkhorn':False })
            K_plus = kernel.fit_transform(Gs_plus)

            kernel = WWL.WWL(param = {'discount':w,'h':2, 'sinkhorn':False })
            K_negative = kernel.fit_transform(Gs_negative)
            v,_ = np.linalg.eigh(K)
            if np.any(v < -10e-8):
                raise ValueError("Not psd")
            K = np.multiply(K_plus, K_negative)
            kernel_hypothesis.Bootstrap(K, function_arguments, B = B)
            out_p_val['wwl'+str(w)].append(kernel_hypothesis.p_values["MMD_u"])




    return pd.DataFrame(out_p_val)




def iteration_weight_sign_attr2(N, B, bg1, bg2, scale2, loc_attr, scale_attr, p2):
    import warnings
    warnings.filterwarnings("ignore")

    def edge_dist( scale ):
        from scipy.stats import uniform
        return np.random.exponential(scale = scale)# uniform.rvs(size=1,  loc = loc , scale = scale)[0]
    def add_weight(G, scale ):
        edge_w = dict()
        for e in G.edges():
            edge_w[e] = edge_dist(scale)
        return edge_w


    prop_w = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    pyramid_L = [2,6,10]
    wl_itr = [2,4]

    out_p_val = dict()
    # out_p_val = {'rw_weight':[],
    #             'rw_signed':[],
    #             #'sp':[],
    #             'rw_binary':[],
    #             'rw_abs':[],
    #             'Bonferroni':[],
    #             'Tensor':[]}

    round_edges = [1,2,3]
    round_nodes = [1,2]

    # for round_edge in round_edges:
    #     for round_node in round_nodes:
    #         out_p_val['sp_n' + str(round_node) + "e" + str(round_edge)] = []
    #         out_p_val['sp_abs_n' + str(round_node) + "e" + str(round_edge)] = []

    #     out_p_val['sp' + "e" + str(round_edge)] = []
    #     out_p_val['sp_abs' +  "e" + str(round_edge)] = []

    #for w in prop_w:
        #out_p_val['propagation' + str(w)] = []
        # out_p_val['wwl' + str(w)] = []

        #out_p_val['propagation_abs' + str(w)] = []
        # out_p_val['wwl_abs' + str(w)] = []
        


    # for L in pyramid_L:
    #     out_p_val['pyramid' + str(L)+'wlab'] = []
    #     out_p_val['pyramid' + str(L)] = []

    for h in wl_itr:
        #out_p_val['wl' + str(h)] = []
        #out_p_val['wloa' + str(h)] = []


        #out_p_val['wl_abs' + str(h)] = []
        #out_p_val['wloa_abs' + str(h)] = []


        for round_node in round_nodes:
            out_p_val['wl_abs' + str(h) + "_n" + str(round_node)] = []
            out_p_val['wloa_abs' + str(h) + "_n" + str(round_node)] = []

            out_p_val['wl' + str(h) + "_n" + str(round_node)] = []
            out_p_val['wloa' + str(h) + "_n" + str(round_node)] = []

    for w in prop_w:
        for round_node in round_nodes:
            out_p_val['wwl_abs' + str(w)+ "_n" + str(round_node)] = []
            out_p_val['wwl' + str(w)+ "_n" + str(round_node)] = []
    


    for _ in tqdm.tqdm(range(N)):
        bg1.Generate()
        bg2.Generate()

        G1 = bg1.Gs.copy()
        G2 = bg2.Gs.copy()

        for G in G1:
            nx.set_edge_attributes(G, add_weight(G, scale = 3000), "weight")
        for G in G2:
            nx.set_edge_attributes(G, add_weight(G, scale = scale2), "weight")

        for G in G1:
            nx.set_node_attributes(G, {i:str(k) for i,k in G.degree}, "label")
        for G in G2:
            nx.set_node_attributes(G, {i:str(k) for i,k in G.degree}, "label")


        for G in G1:
            for e in G.edges():
                if np.random.uniform() <0.35:
                    w = G.edges[e[0], e[1]]['weight']
                    G.edges[e[0], e[1]]['weight'] = -w 

        for G in G2:
            for e in G.edges():
                if np.random.uniform() < p2:
                    w = G.edges[e[0], e[1]]['weight']
                    G.edges[e[0], e[1]]['weight'] = -w 

        for G in G1:
            nx.set_node_attributes(G, {i:np.random.normal(size = (1,), loc = 0.00038, scale= 0.01) for i in range(G.number_of_nodes())}, "attr")
        for G in G2:
            nx.set_node_attributes(G, {i:np.random.normal(size = (1,), loc = loc_attr, scale= scale_attr) for i in range(G.number_of_nodes())}, "attr")
                    
        for G in G1:
            nx.set_edge_attributes(G, {(n1, n2): np.sign(w) for n1, n2, w in G.edges().data('weight')}, "sign")
        for G in G2:
            nx.set_edge_attributes(G, {(n1, n2): np.sign(w) for n1, n2, w in G.edges().data('weight')}, "sign")

        Gs = G1 + G2
        
        Gs_abs = []
        Gs_binary = [nx.from_numpy_array(np.sign(nx.adjacency_matrix(G).todense())) for G in Gs]
        Gs_plus = []
        Gs_negative = []
        for i in range(len(Gs)):
            A = nx.adjacency_matrix(Gs[i]).todense()
            A_plus = A.copy()
            A_plus[A_plus<0] =0
            Gs_plus.append(nx.from_numpy_array(A_plus))
            nx.set_node_attributes(Gs_plus[i], nx.get_node_attributes(Gs[i], 'attr'), "attr")
            nx.set_node_attributes(Gs_plus[i], {j:str(k) for j,k in Gs[i].degree}, "label")


            A_negative = A.copy()
            A_negative[A_negative>0] =0
            A_negative = np.abs(A_negative)
            Gs_negative.append(nx.from_numpy_array(A_negative))
            nx.set_node_attributes(Gs_negative[i], nx.get_node_attributes(Gs[i], 'attr'), "attr")
            nx.set_node_attributes(Gs_negative[i], {j:str(k) for j,k in Gs[i].degree}, "label")


            Gs_abs.append(nx.from_numpy_array(np.abs(A.copy())))
            nx.set_node_attributes(Gs_abs[i], nx.get_node_attributes(Gs[i], 'attr'), "attr")
            nx.set_node_attributes(Gs_abs[i], {j:str(k) for j,k in Gs[i].degree}, "label")


        #print("Graph preperation done...")   
        MMD_functions = [mg.MMD_b, mg.MMD_u, mg.MMD_l]#, mg.MONK_EST]
        kernel_hypothesis = mg.BoostrapMethods(MMD_functions)
        function_arguments = [dict(n = bg1.n, m = bg1.n  ), 
                    dict(n = bg1.n, m = bg1.n ),
                    dict(n = bg1.n, m = bg1.n )]#, 
                    #dict(Q = 3, y1 = Gs[:bg1.n], y2 = Gs[bg1.n:] )]

        p = [np.resize(list(nx.get_node_attributes(G, 'attr').values()), new_shape = (20)) for G in Gs]
        # try:
        # #print("Signed RW")
        #     rw_kernel = rw.RandomWalk(Gs, c = 1e-10, normalize=0, p = p, q = p)
        #     K_sign = rw_kernel.fit_ARKU_edge(r=6, verbose = False, edge_attr='weight', edge_labels=[1.0,-1.0], edge_label_tag = 'sign')
        #     v,_ = np.linalg.eigh(K_sign)
        #     # v[np.abs(v) < 10e-5] = 0
        #     if np.any(v < -10e-8):
        #         raise ValueError("Not psd SIGN EDGE")
        #     kernel_hypothesis.Bootstrap(K_sign, function_arguments, B = B)
        #     out_p_val['rw_signed'].append(kernel_hypothesis.p_values["MMD_u"])
        # except:
        #     continue


        # #print("RW as-is")
        # rw_kernel = rw.RandomWalk(Gs, c = 1e-10, normalize=0, p = p, q = p)
        # K_rw = rw_kernel.fit_ARKU_plus(r=6, verbose = False, edge_attr='weight')
        # v,_ = np.linalg.eigh(K_rw)
        # # v[np.abs(v) < 10e-5] = 0
        # if np.any(v < -10e-8):
        #     raise ValueError("Not psd WEIGHT")
        # kernel_hypothesis.Bootstrap(K_rw, function_arguments, B = B)
        # out_p_val['rw_weight'].append(kernel_hypothesis.p_values["MMD_u"])
        # # print(f'p_value {kernel_hypothesis.p_values["MMD_u"]}')



        # #print("RW absolute value")
        # rw_kernel = rw.RandomWalk(Gs_abs, c = 1e-10, normalize=0, p = p, q = p)
        # K = rw_kernel.fit_ARKU_plus(r=6, verbose = False, edge_attr='weight')
        # v,_ = np.linalg.eigh(K)
        # # v[np.abs(v) < 10e-5] = 0
        # if np.any(v < -10e-8):
        #     raise ValueError("Not psd ARKU")
        # kernel_hypothesis.Bootstrap(K, function_arguments, B = B)
        # out_p_val['rw_abs'].append(kernel_hypothesis.p_values["MMD_u"])
        # # print(f'p_value {kernel_hypothesis.p_values["MMD_u"]}')

        # rw_kernel = rw.RandomWalk(Gs_binary, c = 1e-10, normalize=0, p = p, q = p)
        # K = rw_kernel.fit_ARKU_plus(r=6, verbose = False, edge_attr=None)
        # v,_ = np.linalg.eigh(K)
        # # v[np.abs(v) < 10e-5] = 0
        # if np.any(v < -10e-8):
        #     raise ValueError("Not psd BINARY")
        # kernel_hypothesis.Bootstrap(K, function_arguments, B = B)
        # out_p_val['rw_binary'].append(kernel_hypothesis.p_values["MMD_u"])
        # # print(f'p_value {kernel_hypothesis.p_values["MMD_u"]}')

        
        # #print("Bonferroni")
        # rw_kernel = rw.RandomWalk(Gs_plus, c = 1e-10, normalize=0, p = p, q = p)
        # K1 = rw_kernel.fit_ARKU_plus(r=6, verbose = False, edge_attr='weight')
        # v,_ = np.linalg.eigh(K1)
        # # v[np.abs(v) < 10e-5] = 0
        # if np.any(v < -10e-8):
        #     raise ValueError("Not psd BON 1")
        # kernel_hypothesis.Bootstrap(K1, function_arguments, B = B)
        # pval_1 = kernel_hypothesis.p_values["MMD_u"]
        # rw_kernel = rw.RandomWalk(Gs_negative, c = 1e-10, normalize=0, p = p, q = p)
        # K2 = rw_kernel.fit_ARKU_plus(r=6, verbose = False, edge_attr='weight')
        # v,_ = np.linalg.eigh(K2)
        # # v[np.abs(v) < 10e-5] = 0
        # if np.any(v < -10e-8):
        #     raise ValueError("Not psd BON 2")
        # kernel_hypothesis.Bootstrap(K2, function_arguments, B = B)
        # pval_2 = kernel_hypothesis.p_values["MMD_u"]
        # out_p_val['Bonferroni'].append(np.max((pval_1,pval_2)))
        # #print(f'p_value {pval_1} {pval_2}')

        # #print("Tensor")
        # K = np.multiply(K1,K2)
        # v,_ = np.linalg.eigh(K)
        # # v[np.abs(v) < 10e-5] = 0
        # if np.any(v < -10e-8):
        #     raise ValueError("Not psd TENSOR")
        # kernel_hypothesis.Bootstrap(K, function_arguments, B = B)
        # # print(f'p_value {kernel_hypothesis.p_values["MMD_u"]}')
        # out_p_val['Tensor'].append(kernel_hypothesis.p_values["MMD_u"])

        # for L in pyramid_L:
        #     pm = gk.PyramidMatch(with_labels = False, L = L)
        #     gk_gs = gk.graph_from_networkx(Gs_plus, edge_weight_tag='weight',  node_labels_tag = 'label')
        #     K_plus = pm.fit_transform(gk_gs)

        #     pm = gk.PyramidMatch(with_labels = False, L = L)
        #     gk_gs = gk.graph_from_networkx(Gs_negative, edge_weight_tag='weight',  node_labels_tag = 'label')
        #     K_negative = pm.fit_transform(gk_gs)

        #     K = np.multiply(K_plus, K_negative)
        #     v,_ = np.linalg.eigh(K)
        #     if np.any(v < -10e-8):
        #         raise ValueError("Not psd pyramid")

        #     kernel_hypothesis.Bootstrap(K, function_arguments, B = B)
        #     out_p_val['pyramid' + str(L)].append(kernel_hypothesis.p_values["MMD_u"])   



        # for L in pyramid_L:
        #     pm = gk.PyramidMatch(with_labels = True, L = L)
        #     gk_gs = gk.graph_from_networkx(Gs_plus, edge_weight_tag='weight',  node_labels_tag = 'label')
        #     K_plus = pm.fit_transform(gk_gs)

        #     pm = gk.PyramidMatch(with_labels = True, L = L)
        #     gk_gs = gk.graph_from_networkx(Gs_negative, edge_weight_tag='weight',  node_labels_tag = 'label')
        #     K_negative = pm.fit_transform(gk_gs)

        #     K = np.multiply(K_plus, K_negative)
        #     v,_ = np.linalg.eigh(K)
        #     if np.any(v < -10e-8):
        #         raise ValueError("Not psd pyramid LABEL")

        #     kernel_hypothesis.Bootstrap(K, function_arguments, B = B)
        #     out_p_val['pyramid' + str(L)+'wlab'].append(kernel_hypothesis.p_values["MMD_u"])


        # for w in prop_w:
        #     prop = PropagationAttr(w = w,t_max = 4)
        #     gk_gs = gk.graph_from_networkx(Gs_plus, edge_weight_tag='weight',  node_labels_tag = 'attr')
        #     K_plus = prop.fit_transform(gk_gs)

        #     prop = PropagationAttr(w = w,t_max = 4)
        #     gk_gs = gk.graph_from_networkx(Gs_negative, edge_weight_tag='weight',  node_labels_tag = 'attr')
        #     K_negative = prop.fit_transform(gk_gs)

        #     K = np.multiply(K_plus, K_negative)
        #     v,_ = np.linalg.eigh(K)
        #     if np.any(v < -10e-8):
        #         raise ValueError("Not psd prop")

        #     kernel_hypothesis.Bootstrap(K, function_arguments, B = B)
        #     out_p_val['propagation' + str(w)].append(kernel_hypothesis.p_values["MMD_u"])

        #     prop = PropagationAttr(w = w,t_max = 4)
        #     gk_gs = gk.graph_from_networkx(Gs_abs, edge_weight_tag='weight',  node_labels_tag = 'attr')
        #     K = prop.fit_transform(gk_gs)
        #     kernel_hypothesis.Bootstrap(K, function_arguments, B = B)
        #     out_p_val['propagation_abs' + str(w)].append(kernel_hypothesis.p_values["MMD_u"])

        
        # for round_edge in round_edges:
        #     for round_node in round_nodes:
        #         Gs_sp_plus = prepare_gs_for_sp(Gs_plus, round_node= round_node, round_edge= round_edge)
        #         Gs_sp_negative = prepare_gs_for_sp(Gs_negative, round_node= round_node, round_edge= round_edge)

        #         init_kernel = gk.ShortestPath(normalize=0, with_labels=True)
        #         graph_list = gk.graph_from_networkx(Gs_sp_plus, edge_weight_tag='weight', node_labels_tag= 'label')  # Convert to graphs to Grakel format
        #         K_plus = init_kernel.fit_transform(graph_list)

        #         init_kernel = gk.ShortestPath(normalize=0, with_labels=True)
        #         graph_list = gk.graph_from_networkx(Gs_sp_negative, edge_weight_tag='weight', node_labels_tag= 'label')  # Convert to graphs to Grakel format
        #         K_negative = init_kernel.fit_transform(graph_list)

        #         K = np.multiply(K_plus, K_negative)
        #         v,_ = np.linalg.eigh(K)
        #         # v[np.abs(v) < 10e-5] = 0
        #         if np.any(v < -10e-8):
        #             raise ValueError("Not psd BINARY")
        #         kernel_hypothesis.Bootstrap(K, function_arguments, B = B)
        #         out_p_val['sp_n' + str(round_node) + "e" + str(round_edge)].append(kernel_hypothesis.p_values["MMD_u"])

        #         Gs_abs_sp = prepare_gs_for_sp(Gs_abs, round_node= round_node, round_edge= round_edge)
        #         init_kernel = gk.ShortestPath(normalize=0, with_labels=True)
        #         graph_list = gk.graph_from_networkx(Gs_abs_sp, edge_weight_tag='weight', node_labels_tag= 'label')  # Convert to graphs to Grakel format
        #         K = init_kernel.fit_transform(graph_list)

        #         v,_ = np.linalg.eigh(K)
        #         # v[np.abs(v) < 10e-5] = 0
        #         if np.any(v < -10e-8):
        #             raise ValueError("Not psd BINARY")
        #         kernel_hypothesis.Bootstrap(K, function_arguments, B = B)
        #         out_p_val['sp_abs_n' + str(round_node) + "e" + str(round_edge)].append(kernel_hypothesis.p_values["MMD_u"])



        #     Gs_sp_plus = prepare_gs_for_sp(Gs_plus, round_node= round_node, round_edge= round_edge)
        #     Gs_sp_negative = prepare_gs_for_sp(Gs_negative, round_node= round_node, round_edge= round_edge)

        #     init_kernel = gk.ShortestPath(normalize=0, with_labels=False)
        #     graph_list = gk.graph_from_networkx(Gs_sp_plus, edge_weight_tag='weight', node_labels_tag= 'label')  # Convert to graphs to Grakel format
        #     K_plus = init_kernel.fit_transform(graph_list)

        #     init_kernel = gk.ShortestPath(normalize=0, with_labels=False)
        #     graph_list = gk.graph_from_networkx(Gs_sp_negative, edge_weight_tag='weight', node_labels_tag= 'label')  # Convert to graphs to Grakel format
        #     K_negative = init_kernel.fit_transform(graph_list)

        #     K = np.multiply(K_plus, K_negative)
        #     v,_ = np.linalg.eigh(K)
        #     # v[np.abs(v) < 10e-5] = 0
        #     if np.any(v < -10e-8):
        #         raise ValueError("Not psd BINARY")
        #     kernel_hypothesis.Bootstrap(K, function_arguments, B = B)
        #     out_p_val['sp' +  "e" + str(round_edge)].append(kernel_hypothesis.p_values["MMD_u"])

        #     Gs_abs_sp = prepare_gs_for_sp(Gs_abs, round_node= round_node, round_edge= round_edge)
        #     init_kernel = gk.ShortestPath(normalize=0, with_labels=False)
        #     graph_list = gk.graph_from_networkx(Gs_abs_sp, edge_weight_tag='weight', node_labels_tag= 'label')  # Convert to graphs to Grakel format
        #     K = init_kernel.fit_transform(graph_list)

        #     v,_ = np.linalg.eigh(K)
        #     # v[np.abs(v) < 10e-5] = 0
        #     if np.any(v < -10e-8):
        #         raise ValueError("Not psd BINARY")
        #     kernel_hypothesis.Bootstrap(K, function_arguments, B = B)
        #     out_p_val['sp_abs' + "e" + str(round_edge)].append(kernel_hypothesis.p_values["MMD_u"])



        for h in wl_itr:
            # kernel = [{"name": "weisfeiler_lehman", "n_iter":h}, {"name": "vertex_histogram"}]
            # init_kernel = gk.GraphKernel(kernel= kernel, normalize=0)
            # graph_list = gk.graph_from_networkx(Gs_plus, node_labels_tag='label')  # Convert to graphs to Grakel format
            # K_plus = init_kernel.fit_transform(graph_list)

            # kernel = [{"name": "weisfeiler_lehman", "n_iter":h}, {"name": "vertex_histogram"}]
            # init_kernel = gk.GraphKernel(kernel= kernel, normalize=0)
            # graph_list = gk.graph_from_networkx(Gs_negative, node_labels_tag='label')  # Convert to graphs to Grakel format
            # K_negative = init_kernel.fit_transform(graph_list)

            # K = np.multiply(K_plus, K_negative)
            # v,_ = np.linalg.eigh(K)
            # if np.any(v < -10e-8):
            #     raise ValueError("Not psd wl")

            # kernel_hypothesis.Bootstrap(K, function_arguments, B = B)
            # out_p_val['wl'+str(h)].append(kernel_hypothesis.p_values["MMD_u"])

            # kernel = [{"name": "weisfeiler_lehman", "n_iter":h}, {"name": "vertex_histogram"}]
            # init_kernel = gk.GraphKernel(kernel= kernel, normalize=0)
            # gk_gs = gk.graph_from_networkx(Gs_abs, node_labels_tag='label')
            # K = init_kernel.fit_transform(gk_gs)
            # kernel_hypothesis.Bootstrap(K, function_arguments, B = B)
            # out_p_val['wl_abs' + str(h)].append(kernel_hypothesis.p_values["MMD_u"])

            for round_node in round_nodes:
                Gs_wl_plus = prepare_gs_for_sp(Gs_plus, round_node= round_node, round_edge= 1)
                Gs_wl_negative = prepare_gs_for_sp(Gs_negative, round_node= round_node, round_edge= 1)

                kernel = [{"name": "weisfeiler_lehman", "n_iter":h}, {"name": "vertex_histogram"}]
                init_kernel = gk.GraphKernel(kernel= kernel, normalize=0)
                graph_list = gk.graph_from_networkx(Gs_wl_plus, node_labels_tag= 'label')  # Convert to graphs to Grakel format
                K_plus = init_kernel.fit_transform(graph_list)

                kernel = [{"name": "weisfeiler_lehman", "n_iter":h}, {"name": "vertex_histogram"}]
                init_kernel = gk.GraphKernel(kernel= kernel, normalize=0)
                graph_list = gk.graph_from_networkx(Gs_wl_negative, node_labels_tag= 'label')  # Convert to graphs to Grakel format
                K_negative = init_kernel.fit_transform(graph_list)

                K = np.multiply(K_plus, K_negative)
                v,_ = np.linalg.eigh(K)
                # v[np.abs(v) < 10e-5] = 0
                if np.any(v < -10e-8):
                    raise ValueError("Not psd BINARY")
                kernel_hypothesis.Bootstrap(K, function_arguments, B = B)
                out_p_val['wl' + str(h) + "_n" + str(round_node)].append(kernel_hypothesis.p_values["MMD_u"])


                Gs_abs_wl = prepare_gs_for_sp(Gs_abs, round_node= round_node, round_edge= 1)
                kernel = [{"name": "weisfeiler_lehman", "n_iter":h}, {"name": "vertex_histogram"}]
                init_kernel = gk.GraphKernel(kernel= kernel, normalize=0)
                graph_list = gk.graph_from_networkx(Gs_abs_wl, edge_weight_tag='weight', node_labels_tag= 'label')  # Convert to graphs to Grakel format
                K = init_kernel.fit_transform(graph_list)

                v,_ = np.linalg.eigh(K)
                # v[np.abs(v) < 10e-5] = 0
                if np.any(v < -10e-8):
                    raise ValueError("Not psd BINARY")
                kernel_hypothesis.Bootstrap(K, function_arguments, B = B)
    
                out_p_val['wl_abs' + str(h) + "_n" + str(round_node)].append(kernel_hypothesis.p_values["MMD_u"])

        for h in wl_itr:
            # kernel = [{"name": "WL-OA", "n_iter": h}]
            # init_kernel = gk.GraphKernel(kernel= kernel, normalize=0)
            # graph_list = gk.graph_from_networkx(Gs_plus, node_labels_tag='label')  # Convert to graphs to Grakel format
            # K_plus = init_kernel.fit_transform(graph_list)

            # kernel = [{"name": "WL-OA", "n_iter": h}]
            # init_kernel = gk.GraphKernel(kernel= kernel, normalize=0)
            # graph_list = gk.graph_from_networkx(Gs_negative, node_labels_tag='label')  # Convert to graphs to Grakel format
            # K_negative = init_kernel.fit_transform(graph_list)

            # K = np.multiply(K_plus, K_negative)
            # v,_ = np.linalg.eigh(K)
            # if np.any(v < -10e-8):
            #     raise ValueError("Not psd wloa")
            # kernel_hypothesis.Bootstrap(K, function_arguments, B = B)
            # out_p_val['wloa'+str(h)].append(kernel_hypothesis.p_values["MMD_u"])

            # kernel = [{"name": "WL-OA", "n_iter": h}]
            # init_kernel = gk.GraphKernel(kernel= kernel, normalize=0)
            # gk_gs = gk.graph_from_networkx(Gs_abs, node_labels_tag='label')
            # K = init_kernel.fit_transform(gk_gs)
            # kernel_hypothesis.Bootstrap(K, function_arguments, B = B)
            # out_p_val['wloa_abs' + str(h)].append(kernel_hypothesis.p_values["MMD_u"])

            for round_node in round_nodes:
                Gs_wl_plus = prepare_gs_for_sp(Gs_plus, round_node= round_node, round_edge= 1)
                Gs_wl_negative = prepare_gs_for_sp(Gs_negative, round_node= round_node, round_edge= 1)

                kernel = [{"name": "WL-OA", "n_iter": h}]
                init_kernel = gk.GraphKernel(kernel= kernel, normalize=0)
                graph_list = gk.graph_from_networkx(Gs_wl_plus, node_labels_tag= 'label')  # Convert to graphs to Grakel format
                K_plus = init_kernel.fit_transform(graph_list)

                kernel = [{"name": "WL-OA", "n_iter": h}]
                init_kernel = gk.GraphKernel(kernel= kernel, normalize=0)
                graph_list = gk.graph_from_networkx(Gs_wl_negative, node_labels_tag= 'label')  # Convert to graphs to Grakel format
                K_negative = init_kernel.fit_transform(graph_list)

                K = np.multiply(K_plus, K_negative)
                v,_ = np.linalg.eigh(K)
                # v[np.abs(v) < 10e-5] = 0
                if np.any(v < -10e-8):
                    raise ValueError("Not psd BINARY")
                kernel_hypothesis.Bootstrap(K, function_arguments, B = B)
                out_p_val['wloa' + str(h) + "_n" + str(round_node)].append(kernel_hypothesis.p_values["MMD_u"])


                Gs_abs_wl = prepare_gs_for_sp(Gs_abs, round_node= round_node, round_edge= 1)
                kernel = [{"name": "WL-OA", "n_iter": h}]
                init_kernel = gk.GraphKernel(kernel= kernel, normalize=0)
                graph_list = gk.graph_from_networkx(Gs_abs_wl, edge_weight_tag='weight', node_labels_tag= 'label')  # Convert to graphs to Grakel format
                K = init_kernel.fit_transform(graph_list)

                v,_ = np.linalg.eigh(K)
                # v[np.abs(v) < 10e-5] = 0
                if np.any(v < -10e-8):
                    raise ValueError("Not psd BINARY")
                kernel_hypothesis.Bootstrap(K, function_arguments, B = B)
    
                out_p_val['wloa_abs' + str(h) + "_n" + str(round_node)].append(kernel_hypothesis.p_values["MMD_u"])


        for w in prop_w:
            # kernel = WWL.WWL(param = {'discount':w,'h':2, 'sinkhorn':False })
            # K_plus = kernel.fit_transform(Gs_plus)

            # kernel = WWL.WWL(param = {'discount':w,'h':2, 'sinkhorn':False })
            # K_negative = kernel.fit_transform(Gs_negative)
            # v,_ = np.linalg.eigh(K)
            # if np.any(v < -10e-8):
            #     raise ValueError("Not psd")
            # K = np.multiply(K_plus, K_negative)
            # kernel_hypothesis.Bootstrap(K, function_arguments, B = B)
            # out_p_val['wwl'+str(w)].append(kernel_hypothesis.p_values["MMD_u"])

            # kernel = WWL.WWL(param = {'discount':w,'h':2, 'sinkhorn':False })
            # K = kernel.fit_transform(Gs_abs)
            # kernel_hypothesis.Bootstrap(K, function_arguments, B = B)
            # out_p_val['wwl_abs'+str(w)].append(kernel_hypothesis.p_values["MMD_u"])

            for round_node in round_nodes:
                Gs_wl_plus = prepare_gs_for_sp(Gs_plus, round_node= round_node, round_edge= 1)
                Gs_wl_negative = prepare_gs_for_sp(Gs_negative, round_node= round_node, round_edge= 1)

                kernel = WWL.WWL(param = {'discount':w,'h':2, 'sinkhorn':False })
                K_plus = kernel.fit_transform(Gs_wl_plus)

                kernel = WWL.WWL(param = {'discount':w,'h':2, 'sinkhorn':False })
                K_negative = kernel.fit_transform(Gs_wl_negative)

                K = np.multiply(K_plus, K_negative)
                v,_ = np.linalg.eigh(K)
                # v[np.abs(v) < 10e-5] = 0
                if np.any(v < -10e-8):
                    raise ValueError("Not psd BINARY")
                kernel_hypothesis.Bootstrap(K, function_arguments, B = B)
                out_p_val['wwl' + str(w)+ "_n" + str(round_node)].append(kernel_hypothesis.p_values["MMD_u"])


                Gs_abs_wl = prepare_gs_for_sp(Gs_abs, round_node= round_node, round_edge= 1)
                kernel = WWL.WWL(param = {'discount':w,'h':2, 'sinkhorn':False })
                K = kernel.fit_transform(Gs_abs_wl)

                v,_ = np.linalg.eigh(K)
                # v[np.abs(v) < 10e-5] = 0
                if np.any(v < -10e-8):
                    raise ValueError("Not psd BINARY")
                kernel_hypothesis.Bootstrap(K, function_arguments, B = B)
        
                out_p_val['wwl_abs' + str(w)+ "_n" + str(round_node)].append(kernel_hypothesis.p_values["MMD_u"])



    return pd.DataFrame(out_p_val)



def prepare_gs_for_sp(Gs, round_edge, round_node):
    
    Gs_new  = []

    max_weight = 0
    for i in range(len(Gs)):
        max_weight_i = np.max([np.abs(w[2]) for w in Gs[i].edges(data = 'weight')])
        if max_weight_i > max_weight:
            max_weight = np.abs(max_weight_i)

    max_return = 0

    for i in range(len(Gs)):
        max_return_i = np.max([w[1] for w in Gs[i].nodes(data = 'attr')])
        if max_weight_i > max_return:
            max_return = np.abs(max_return_i)
            
    for i in range(len(Gs)):

        Gs_new.append(nx.from_numpy_array(np.abs(np.round(nx.adjacency_matrix(Gs[i]).todense()/max_weight, round_edge))))
        nx.set_node_attributes(Gs_new[i], {v:str(np.round(w[0]/max_return,round_node)) for v, w in nx.get_node_attributes(Gs[0], 'attr').items()}, 'label')

    return Gs_new

def iteration_weight_sign_attr_sp2(N, B, bg1, bg2, scale2, loc_attr, scale_attr, p2):

    def edge_dist( scale ):
        from scipy.stats import uniform
        return np.random.exponential(scale = scale)# uniform.rvs(size=1,  loc = loc , scale = scale)[0]
    def add_weight(G, scale ):
        edge_w = dict()
        for e in G.edges():
            edge_w[e] = edge_dist(scale)
        return edge_w


    round_edges = [1,2,3]
    round_nodes = [1,2]

    out_p_val = dict()
    for round_edge in round_edges:
        for round_node in round_nodes:
            out_p_val['sp_n' + str(round_node) + "e" + str(round_edge)] = []
            out_p_val['sp_abs_n' + str(round_node) + "e" + str(round_edge)] = []

        out_p_val['sp' + "e" + str(round_edge)] = []
        out_p_val['sp_abs' +  "e" + str(round_edge)] = []


    print(out_p_val.keys())
    for _ in tqdm.tqdm(range(N)):
        bg1.Generate()
        bg2.Generate()

        G1 = bg1.Gs.copy()
        G2 = bg2.Gs.copy()

        for G in G1:
            nx.set_edge_attributes(G, add_weight(G, scale = 3000), "weight")
        for G in G2:
            nx.set_edge_attributes(G, add_weight(G, scale = scale2), "weight")

        for G in G1:
            nx.set_node_attributes(G, {i:str(k) for i,k in G.degree}, "label")
        for G in G2:
            nx.set_node_attributes(G, {i:str(k) for i,k in G.degree}, "label")


        for G in G1:
            for e in G.edges():
                if np.random.uniform() <0.35:
                    w = G.edges[e[0], e[1]]['weight']
                    G.edges[e[0], e[1]]['weight'] = -w 

        for G in G2:
            for e in G.edges():
                if np.random.uniform() < p2:
                    w = G.edges[e[0], e[1]]['weight']
                    G.edges[e[0], e[1]]['weight'] = -w 

        for G in G1:
            nx.set_node_attributes(G, {i:np.random.normal(size = (1,), loc = 0.00038, scale= 0.01) for i in range(G.number_of_nodes())}, "attr")
        for G in G2:
            nx.set_node_attributes(G, {i:np.random.normal(size = (1,), loc = loc_attr, scale= scale_attr) for i in range(G.number_of_nodes())}, "attr")
                    
        for G in G1:
            nx.set_edge_attributes(G, {(n1, n2): np.sign(w) for n1, n2, w in G.edges().data('weight')}, "sign")
        for G in G2:
            nx.set_edge_attributes(G, {(n1, n2): np.sign(w) for n1, n2, w in G.edges().data('weight')}, "sign")

        Gs = G1 + G2
        
        Gs_abs = []
        Gs_binary = [nx.from_numpy_array(np.sign(nx.adjacency_matrix(G).todense())) for G in Gs]
        Gs_plus = []
        Gs_negative = []
        for i in range(len(Gs)):
            A = nx.adjacency_matrix(Gs[i]).todense()
            A_plus = A.copy()
            A_plus[A_plus<0] =0
            Gs_plus.append(nx.from_numpy_array(A_plus))
            nx.set_node_attributes(Gs_plus[i], nx.get_node_attributes(Gs[i], 'attr'), "attr")
            nx.set_node_attributes(Gs_plus[i], {j:str(k) for j,k in Gs[i].degree}, "label")


            A_negative = A.copy()
            A_negative[A_negative>0] =0
            A_negative = np.abs(A_negative)
            Gs_negative.append(nx.from_numpy_array(A_negative))
            nx.set_node_attributes(Gs_negative[i], nx.get_node_attributes(Gs[i], 'attr'), "attr")
            nx.set_node_attributes(Gs_negative[i], {j:str(k) for j,k in Gs[i].degree}, "label")


            Gs_abs.append(nx.from_numpy_array(np.abs(A.copy())))
            nx.set_node_attributes(Gs_abs[i], nx.get_node_attributes(Gs[i], 'attr'), "attr")
            nx.set_node_attributes(Gs_abs[i], {j:str(k) for j,k in Gs[i].degree}, "label")


        #print("Graph preperation done...")   
        MMD_functions = [mg.MMD_b, mg.MMD_u, mg.MMD_l]#, mg.MONK_EST]
        kernel_hypothesis = mg.BoostrapMethods(MMD_functions)
        function_arguments = [dict(n = bg1.n, m = bg1.n  ), 
                    dict(n = bg1.n, m = bg1.n ),
                    dict(n = bg1.n, m = bg1.n )]#, 
                    #dict(Q = 3, y1 = Gs[:bg1.n], y2 = Gs[bg1.n:] )]

        p = [np.resize(list(nx.get_node_attributes(G, 'attr').values()), new_shape = (20)) for G in Gs]
        


        for round_edge in round_edges:
            for round_node in round_nodes:
                Gs_sp_plus = prepare_gs_for_sp(Gs_plus, round_node= round_node, round_edge= round_edge)
                Gs_sp_negative = prepare_gs_for_sp(Gs_negative, round_node= round_node, round_edge= round_edge)

                init_kernel = gk.ShortestPath(normalize=0, with_labels=True)
                graph_list = gk.graph_from_networkx(Gs_sp_plus, edge_weight_tag='weight', node_labels_tag= 'label')  # Convert to graphs to Grakel format
                K_plus = init_kernel.fit_transform(graph_list)

                init_kernel = gk.ShortestPath(normalize=0, with_labels=True)
                graph_list = gk.graph_from_networkx(Gs_sp_negative, edge_weight_tag='weight', node_labels_tag= 'label')  # Convert to graphs to Grakel format
                K_negative = init_kernel.fit_transform(graph_list)

                K = np.multiply(K_plus, K_negative)
                v,_ = np.linalg.eigh(K)
                # v[np.abs(v) < 10e-5] = 0
                if np.any(v < -10e-8):
                    raise ValueError("Not psd BINARY")
                kernel_hypothesis.Bootstrap(K, function_arguments, B = B)
                out_p_val['sp_n' + str(round_node) + "e" + str(round_edge)].append(kernel_hypothesis.p_values["MMD_u"])

                Gs_abs_sp = prepare_gs_for_sp(Gs_abs, round_node= round_node, round_edge= round_edge)
                init_kernel = gk.ShortestPath(normalize=0, with_labels=True)
                graph_list = gk.graph_from_networkx(Gs_abs_sp, edge_weight_tag='weight', node_labels_tag= 'label')  # Convert to graphs to Grakel format
                K = init_kernel.fit_transform(graph_list)

                v,_ = np.linalg.eigh(K)
                # v[np.abs(v) < 10e-5] = 0
                if np.any(v < -10e-8):
                    raise ValueError("Not psd BINARY")
                kernel_hypothesis.Bootstrap(K, function_arguments, B = B)
                out_p_val['sp_abs_n' + str(round_node) + "e" + str(round_edge)].append(kernel_hypothesis.p_values["MMD_u"])



            Gs_sp_plus = prepare_gs_for_sp(Gs_plus, round_node= round_node, round_edge= round_edge)
            Gs_sp_negative = prepare_gs_for_sp(Gs_negative, round_node= round_node, round_edge= round_edge)

            init_kernel = gk.ShortestPath(normalize=0, with_labels=False)
            graph_list = gk.graph_from_networkx(Gs_sp_plus, edge_weight_tag='weight', node_labels_tag= 'label')  # Convert to graphs to Grakel format
            K_plus = init_kernel.fit_transform(graph_list)

            init_kernel = gk.ShortestPath(normalize=0, with_labels=False)
            graph_list = gk.graph_from_networkx(Gs_sp_negative, edge_weight_tag='weight', node_labels_tag= 'label')  # Convert to graphs to Grakel format
            K_negative = init_kernel.fit_transform(graph_list)

            K = np.multiply(K_plus, K_negative)
            v,_ = np.linalg.eigh(K)
            # v[np.abs(v) < 10e-5] = 0
            if np.any(v < -10e-8):
                raise ValueError("Not psd BINARY")
            kernel_hypothesis.Bootstrap(K, function_arguments, B = B)
            out_p_val['sp' +  "e" + str(round_edge)].append(kernel_hypothesis.p_values["MMD_u"])

            Gs_abs_sp = prepare_gs_for_sp(Gs_abs, round_node= round_node, round_edge= round_edge)
            init_kernel = gk.ShortestPath(normalize=0, with_labels=False)
            graph_list = gk.graph_from_networkx(Gs_abs_sp, edge_weight_tag='weight', node_labels_tag= 'label')  # Convert to graphs to Grakel format
            K = init_kernel.fit_transform(graph_list)

            v,_ = np.linalg.eigh(K)
            # v[np.abs(v) < 10e-5] = 0
            if np.any(v < -10e-8):
                raise ValueError("Not psd BINARY")
            kernel_hypothesis.Bootstrap(K, function_arguments, B = B)
            out_p_val['sp_abs' + "e" + str(round_edge)].append(kernel_hypothesis.p_values["MMD_u"])



    return pd.DataFrame(out_p_val)



def roc(mmd_info, path):

    N = float(mmd_info.shape[0])
    # Store outcome in a data
    df = pd.DataFrame()
    alphas = np.linspace(0.001, 0.99, 999)
    for alpha in alphas:
        
        # type II error is the case when be p_val > alpha so power is 1 - #(p_val>alpha)/N <-> (N - #(p_val>alpha))/N <-> #(p_val<alpha)/N
        # power of MMD tests (including distribution free test)
        power_mmd = dict()

        for key in mmd_info.columns:
            #print(f'{key} pvaalue {p_values[key]}')
            power_mmd[key] = (np.array(mmd_info[key]) < alpha).sum()/float(N)

        # Store the run information in a dataframe,
        tmp = pd.DataFrame({'alpha':alpha}, index = [0])
        for key, v in power_mmd.items():
            tmp[key] = v


        # add to the main data frame
        df = pd.concat((df,tmp), ignore_index=True)

    # Save the dataframe at each iteration each such that if out-of-memory or time-out happen we at least have some of the information.
    with open(path, 'wb') as f:
        pickle.dump(df, f)

    return None


if __name__ == '__main__':


    n_1 = 20
    n_2 = 20
    nnode_1 = 20
    nnode_2 = 20




    # for m, s, k2 in [(0.00038, 0.01, 4), (0.0002, 0.01, 4), (0.00038, 0.015, 4), (0.0002, 0.015, 4),
    #                 (0.00038, 0.01, 3.5), (0.0002, 0.01, 3.5), (0.00038, 0.015, 3.5), (0.0002, 0.015, 3.5)]:
    #         bg1 = mg.BinomialGraphs(n_1, nnode_1, k = 4, fullyConnected = True, l = 'degreelabels')
    #         bg2 = mg.BinomialGraphs(n_2, nnode_2, k = k2, fullyConnected = True, l = 'degreelabels')
    #         bg1.Generate()
    #         bg2.Generate()
    #         n_worker = 10
    #         with Pool(n_worker) as pool:

    #             L = pool.starmap(iteration_weight_attr, [( 150,
    #                                             10000,
    #                                             bg1, 
    #                                             bg2, 
    #                                             3500,
    #                                             m, 
    #                                             s) for _ in range(n_worker) ])
            
    #         mmd_info = pd.concat(L)
    #         roc(mmd_info, f'weight_attr_test_n1_{n_1}_n_1_{n_2}_v_1_{nnode_1}_v_2_{nnode_2}_s2_{3500}_k1_{4}_k2_{k2}_m_{m}_s_{s}.pkl')



    # for p2, k2 in [(0.3 ,4), (0.3, 3.5), (0.4 ,4), (0.4, 3.5)]:
    #     bg1 = mg.BinomialGraphs(n_1, nnode_1, k = 4, fullyConnected = True, l = 'degreelabels')
    #     bg2 = mg.BinomialGraphs(n_2, nnode_2, k = k2, fullyConnected = True, l = 'degreelabels')
    #     bg1.Generate()
    #     bg2.Generate()
    #     n_worker = 10
    #     with Pool(n_worker) as pool:

    #         L = pool.starmap(iteration_weight_sign, [( 150,
    #                                         10000,
    #                                         bg1, 
    #                                         bg2, 
    #                                         3500,
    #                                         p2) for _ in range(n_worker) ])
        
    #     mmd_info = pd.concat(L)
    #     roc(mmd_info, f'weight_sign_test_n1_{n_1}_n_1_{n_2}_v_1_{nnode_1}_v_2_{nnode_2}_s2_{3500}_k1_{4}_k2_{k2}_p1_{0.35}p2_{p2}.pkl')


    # for p2, k2, loc_attr, scale_attr, wei in [(0.3, 3.5, 0.0002, 0.012, 3500),
    #                                           (0.35, 4, 0.00038, 0.015, 3500),
    #                                           (0.3, 4, 0.00038, 0.01, 3000),
    #                                           (0.4, 4, 0.00038, 0.01, 3000),
    #                                           (0.3, 3.5, 0.00038, 0.01, 3000),
    #                                           (0.4, 3.5, 0.00038, 0.01, 3000)]:
    #     bg1 = mg.BinomialGraphs(n_1, nnode_1, k = 4, fullyConnected = True, l = 'degreelabels')
    #     bg2 = mg.BinomialGraphs(n_2, nnode_2, k = k2, fullyConnected = True, l = 'degreelabels')
    #     bg1.Generate()
    #     bg2.Generate()
    #     n_worker = 10
    #     with Pool(n_worker) as pool:

    #         L = pool.starmap(iteration_weight_sign_attr2, [( 150,
    #                                         10000,
    #                                         bg1, 
    #                                         bg2, 
    #                                         wei,
    #                                         loc_attr,
    #                                         scale_attr,
    #                                         p2) for _ in range(n_worker) ])
        
    #     mmd_info = pd.concat(L)
    #     roc(mmd_info, f'weight_sign_attr_test_with_abs_n1_{n_1}_n_1_{n_2}_v_1_{nnode_1}_v_2_{nnode_2}_s2_{wei}_k1_{4}_k2_{k2}_p1_{0.35}_p2_{p2}_l2_{loc_attr}_s2_{scale_attr}.pkl')
    
    
    for p2, k2, loc_attr, scale_attr, wei in [(0.3, 3.9, 0.0002, 0.012, 3500),
                                              (0.35, 4, 0.00038, 0.015, 3000),
                                              (0.3, 4, 0.00038, 0.01, 3000),
                                              (0.4, 4, 0.00038, 0.01, 3000)]:
        bg1 = mg.BinomialGraphs(n_1, nnode_1, k = 4, fullyConnected = True, l = 'degreelabels')
        bg2 = mg.BinomialGraphs(n_2, nnode_2, k = k2, fullyConnected = True, l = 'degreelabels')
        bg1.Generate()
        bg2.Generate()


        n_worker = 6
        with Pool(n_worker) as pool:

            L = pool.starmap(iteration_weight_sign_attr2, [( 200,
                                            10000,
                                            bg1, 
                                            bg2, 
                                            wei,
                                            loc_attr,
                                            scale_attr,
                                            p2) for _ in range(n_worker) ])
        
        mmd_info = pd.concat(L)
        roc(mmd_info, f'wl_weight_sign_attr_test_with_abs_n1_{n_1}_n_1_{n_2}_v_1_{nnode_1}_v_2_{nnode_2}_s2_{wei}_k1_{4}_k2_{k2}_p1_{0.35}_p2_{p2}_l2_{loc_attr}_s2_{scale_attr}.pkl')
