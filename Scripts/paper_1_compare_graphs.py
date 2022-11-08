from operator import truediv
from numpy.lib.utils import info
import pandas as pd

import time
import datetime
import importlib
import os, sys
import tqdm
# from pandas_datareader import data
import networkx as nx
sys.path.insert(0, 'C:/Users/User/Code/MMD_Graph_Diversification')
import pickle
import grakel as gk

from sklearn.covariance import graphical_lasso, GraphicalLasso, GraphicalLassoCV


from sklearn.covariance import GraphicalLassoCV
from sklearn.preprocessing import StandardScaler

import warnings


from multiprocessing import Pool, freeze_support



sys.path.insert(0, 'C:/Users/User/Code/MMD_Graph_Diversification/myKernels')
from myKernels import RandomWalk as rw
import MMDforGraphs as mg
import WL
import GNTK
import GraphStatKernel
import WWL
import sp




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

def prepare_gs_for_sp(Gs, round_edge, round_node):
    
    Gs_new  = []

    max_weight = 0
    for i in range(len(Gs)):
        try:
            max_weight_i = np.max([np.abs(w[2]) for w in Gs[i].edges(data = 'weight')])
        except:
            max_weight_i = 0
        if max_weight_i > max_weight:
            max_weight = np.abs(max_weight_i)

    if max_weight == 0:
        max_weight = 1

    max_return = 0

    for i in range(len(Gs)):
        max_return_i = np.max([w[1] for w in Gs[i].nodes(data = 'attr')])
        if max_weight_i > max_return:
            max_return = np.abs(max_return_i)
            
    for i in range(len(Gs)):

        Gs_new.append(nx.from_numpy_array(np.abs(np.round(nx.adjacency_matrix(Gs[i]).todense()/max_weight, round_edge))))
        nx.set_node_attributes(Gs_new[i], {v:str(np.round(w[0]/max_return,round_node)) for v, w in nx.get_node_attributes(Gs[0], 'attr').items()}, 'label')

    return Gs_new

def graph_test(data_dict, study, transform, scale, n,graph_name, ptype, B, edge_attr = 'weight',day_step = 1, graph_label = None, do_tensor = False, kernel_params = None):
    """
    n - number of samples
    data_dict - data dictionary output from Generate_graphs_case_1
    graph_name - which graphs graph_dict or graph_dict2 (with self loops)
    ptype - portfolio tyoe, uniform, sharpe or gmv
    B - nr bootstraps
    edge_attr - If weight, then the rw kernel uses weights
    day_step - "Thinning" of the sample
    graph_label - if none, no labels, if signed then edge labels
    weight_fun - if abs then then all weights will be set as the absolute value
    """

    # nr_splits = 3
    m = n

    graph_dict = data_dict[graph_name]
    sector_name = data_dict['sector']
    dates = data_dict['dates']
    print(sector_name)

    esg_return_df = pd.DataFrame()
    total_length = len(graph_dict[0])
    pbar = tqdm.tqdm( total=len(list(range(n, total_length, day_step)))*1, desc= f'{study} {graph_name} {graph_label} {ptype}')
    for group_1 in [0]:#range(nr_splits):
        for group_2 in [2]:#range(group_1+1, nr_splits):

            for i in range(n, total_length, day_step):
                # print(i-cnt)n

                if graph_name == 'cov_dict':
                    Gs = [ nx.from_numpy_array(1e4*graph_dict[group_1][s]) for s in range(i-n, i )] + [ nx.from_numpy_array(1e4*graph_dict[group_2][s]) for s in range(i-n, i )]
                else:
                    Gs = [ graph_dict[group_1][s] for s in range(i-n, i )] + [ graph_dict[group_2][s] for s in range(i-n, i )]
                

                # rw nr eigenvalues
                r = np.min((6, Gs[0].number_of_nodes()-1))

                # get attributes
                if ptype is None:
                    p = None
                    q = None
                elif ptype == 'return':
                    if graph_label == 'rw':
                        p = np.vstack(([ data_dict['return_dict'][group_1][s]*100 for s in range(i-n, i )],[ data_dict['return_dict'][group_2][s]*100  for s in range(i-n, i )]))
                    else:
                        p = np.vstack(([ data_dict['return_dict'][group_1][s] for s in range(i-n, i )],[ data_dict['return_dict'][group_2][s]  for s in range(i-n, i )]))
                    q = p.copy()
                else:
                    p = np.vstack(([ np.ones(Gs[0].number_of_nodes())/float(Gs[0].number_of_nodes()) for s in range(i-n, i )],[ np.ones(Gs[0].number_of_nodes())/float(Gs[0].number_of_nodes()) for s in range(i-n, i )]))
                    q = p.copy()

                for k in range(len(Gs)):
                    nx.set_node_attributes(Gs[k], {j:[p[k,j]] for j in range(Gs[k].number_of_nodes())}, 'attr')
                    nx.set_node_attributes(Gs[k], {j:str(k) for j,k in Gs[k].degree}, "label")

                if do_tensor:
                    Gs_plus = []
                    Gs_negative = []
                    for k in range(len(Gs)):
                        A = nx.adjacency_matrix(Gs[k]).todense()
                        A_plus = A.copy()
                        A_plus[A_plus<0] =0
                        Gs_plus.append(nx.from_numpy_array(A_plus))
                        nx.set_node_attributes(Gs_plus[k], nx.get_node_attributes(Gs[k], 'attr'), "attr")
                        nx.set_node_attributes(Gs_plus[k], {j:str(k) for j,k in Gs[k].degree}, "label")


                        A_negative = A.copy()
                        A_negative[A_negative>0] =0
                        A_negative = np.abs(A_negative)
                        Gs_negative.append(nx.from_numpy_array(A_negative))
                        nx.set_node_attributes(Gs_negative[k], nx.get_node_attributes(Gs[k], 'attr'), "attr")
                        nx.set_node_attributes(Gs_negative[k], {j:str(k) for j,k in Gs[k].degree}, "label")

                Gs_abs = []
                for k in range(len(Gs)):
                    A = nx.adjacency_matrix(Gs[k]).todense()
                    Gs_abs.append(nx.from_numpy_array(np.abs(A.copy())))
                    nx.set_node_attributes(Gs_abs[k], nx.get_node_attributes(Gs[k], 'attr'), "attr")
                    nx.set_node_attributes(Gs_abs[k], {j:str(k) for j,k in Gs[k].degree}, "label")


                if graph_label  == 'rw':
                    calc_ok = False
                    c_new = kernel_params['c']
                    while not calc_ok:
                        calc_ok = True
                        try:
                            rw_kernel = rw.RandomWalk(Gs, c = c_new, normalize=0, p=p, q = q)
                            K = rw_kernel.fit_ARKU_plus(r = r, normalize_adj=False, verbose=False, edge_attr = edge_attr)

                            v,_ = np.linalg.eigh(K)
                            # v[np.abs(v) < 10e-5] = 0
                            if np.any(v < -10e-12):
                                raise ValueError("Not psd")
                        except:
                            calc_ok = False
                            c_new = c_new*0.8
                            print(f'{study} {graph_name} {graph_label} {ptype} {dates[i]} new c is {c_new}')
                
                if graph_label  == 'rw_scaled':
                
                    for idx_g in range(len(Gs)):
                        scale_edge = np.std([w[2] for w in Gs[idx_g].edges(data = 'weight')])
                        nx.set_edge_attributes(Gs[idx_g],{(w[0], w[1]): w[2]/scale_edge for w in Gs[idx_g].edges(data = 'weight')}, 'weight')

                        scale_node = np.std([w[1][0] for w in Gs[idx_g].nodes(data = 'attr')])
                        nx.set_node_attributes(Gs[idx_g],{w[0]: w[1]/scale_node for w in Gs[idx_g].nodes(data = 'attr')}, 'attr')

                    calc_ok = False
                    c_new = kernel_params['c']
                    while not calc_ok:
                        calc_ok = True
                        try:
                            rw_kernel = rw.RandomWalk(Gs, c = c_new, normalize=0, p=p, q = q)
                            K = rw_kernel.fit_ARKU_plus(r = r, normalize_adj=False, verbose=False, edge_attr = edge_attr)

                            v,_ = np.linalg.eigh(K)
                            # v[np.abs(v) < 10e-5] = 0
                            if np.any(v < -10e-12):
                                raise ValueError("Not psd")
                        except:
                            calc_ok = False
                            c_new = c_new*0.8
                            print(f'{study} {graph_name} {graph_label} {ptype} {dates[i]} new c is {c_new}')
                elif graph_label == 'wl':
                    if do_tensor:
                        kernel = [{"name": "weisfeiler_lehman", "n_iter":kernel_params['h']}, {"name": "vertex_histogram"}]
                        init_kernel = gk.GraphKernel(kernel= kernel, normalize=0)
                        graph_list = gk.graph_from_networkx(Gs_plus, node_labels_tag='label')  # Convert to graphs to Grakel format
                        K_plus = init_kernel.fit_transform(graph_list)

                        kernel = [{"name": "weisfeiler_lehman", "n_iter":kernel_params['h']}, {"name": "vertex_histogram"}]
                        init_kernel = gk.GraphKernel(kernel= kernel, normalize=0)
                        graph_list = gk.graph_from_networkx(Gs_negative, node_labels_tag='label')  # Convert to graphs to Grakel format
                        K_negative = init_kernel.fit_transform(graph_list)

                        K = np.multiply(K_plus, K_negative)

                    else:
                        kernel = [{"name": "weisfeiler_lehman", "n_iter":kernel_params['h']}, {"name": "vertex_histogram"}]
                        init_kernel = gk.GraphKernel(kernel= kernel, normalize=0)
                        graph_list = gk.graph_from_networkx(Gs, node_labels_tag='label')  # Convert to graphs to Grakel format
                        K = init_kernel.fit_transform(graph_list)
                
                elif graph_label == 'wl_bin':
                    if do_tensor:
                        
                        Gs_plus = prepare_gs_for_sp(Gs_plus, round_node= kernel_params['round_node'], round_edge= 2)
                        Gs_negative = prepare_gs_for_sp(Gs_negative, round_node= kernel_params['round_node'], round_edge=2)

                        kernel = [{"name": "weisfeiler_lehman", "n_iter":kernel_params['h']}, {"name": "vertex_histogram"}]
                        init_kernel = gk.GraphKernel(kernel= kernel, normalize=0)
                        graph_list = gk.graph_from_networkx(Gs_plus, node_labels_tag='label')  # Convert to graphs to Grakel format
                        K_plus = init_kernel.fit_transform(graph_list)

                        kernel = [{"name": "weisfeiler_lehman", "n_iter":kernel_params['h']}, {"name": "vertex_histogram"}]
                        init_kernel = gk.GraphKernel(kernel= kernel, normalize=0)
                        graph_list = gk.graph_from_networkx(Gs_negative, node_labels_tag='label')  # Convert to graphs to Grakel format
                        K_negative = init_kernel.fit_transform(graph_list)

                        K = np.multiply(K_plus, K_negative)

                    else:
                        Gs = prepare_gs_for_sp(Gs, round_node= kernel_params['round_node'], round_edge= 2)
                        kernel = [{"name": "weisfeiler_lehman", "n_iter":kernel_params['h']}, {"name": "vertex_histogram"}]
                        init_kernel = gk.GraphKernel(kernel= kernel, normalize=0)
                        graph_list = gk.graph_from_networkx(Gs, node_labels_tag='label')  # Convert to graphs to Grakel format
                        K = init_kernel.fit_transform(graph_list)

                elif graph_label == 'wloa':
                    if do_tensor:
                        kernel = [{"name": "WL-OA", "n_iter": kernel_params['h']}]
                        init_kernel = gk.GraphKernel(kernel= kernel, normalize=0)
                        graph_list = gk.graph_from_networkx(Gs_plus, node_labels_tag='label')  # Convert to graphs to Grakel format
                        K_plus = init_kernel.fit_transform(graph_list)

                        kernel = [{"name": "WL-OA", "n_iter": kernel_params['h']}]
                        init_kernel = gk.GraphKernel(kernel= kernel, normalize=0)
                        graph_list = gk.graph_from_networkx(Gs_negative, node_labels_tag='label')  # Convert to graphs to Grakel format
                        K_negative = init_kernel.fit_transform(graph_list)

                        K = np.multiply(K_plus, K_negative)
                    else:
                        kernel = [{"name": "WL-OA", "n_iter": kernel_params['h']}]
                        init_kernel = gk.GraphKernel(kernel= kernel, normalize=0)
                        graph_list = gk.graph_from_networkx(Gs, node_labels_tag='label')  # Convert to graphs to Grakel format
                        K = init_kernel.fit_transform(graph_list)

                elif graph_label == 'wloa_bin':
                    if do_tensor:

                        Gs_plus = prepare_gs_for_sp(Gs_plus, round_node= kernel_params['round_node'], round_edge= 2)
                        Gs_negative = prepare_gs_for_sp(Gs_negative, round_node= kernel_params['round_node'], round_edge=2)

                        kernel = [{"name": "WL-OA", "n_iter": kernel_params['h']}]
                        init_kernel = gk.GraphKernel(kernel= kernel, normalize=0)
                        graph_list = gk.graph_from_networkx(Gs_plus, node_labels_tag='label')  # Convert to graphs to Grakel format
                        K_plus = init_kernel.fit_transform(graph_list)

                        kernel = [{"name": "WL-OA", "n_iter": kernel_params['h']}]
                        init_kernel = gk.GraphKernel(kernel= kernel, normalize=0)
                        graph_list = gk.graph_from_networkx(Gs_negative, node_labels_tag='label')  # Convert to graphs to Grakel format
                        K_negative = init_kernel.fit_transform(graph_list)

                        K = np.multiply(K_plus, K_negative)
                    else:
                        Gs = prepare_gs_for_sp(Gs, round_node= kernel_params['round_node'], round_edge= 2)
                        kernel = [{"name": "WL-OA", "n_iter": kernel_params['h']}]
                        init_kernel = gk.GraphKernel(kernel= kernel, normalize=0)
                        graph_list = gk.graph_from_networkx(Gs, node_labels_tag='label')  # Convert to graphs to Grakel format
                        K = init_kernel.fit_transform(graph_list)
                elif graph_label == "wwl":

                    if do_tensor:
                        kernel = WWL.WWL(param = {'discount':kernel_params['w'],'h':kernel_params['h'], 'sinkhorn':False })
                        K_plus = kernel.fit_transform(Gs_plus)

                        kernel = WWL.WWL(param = {'discount':kernel_params['w'],'h':kernel_params['h'], 'sinkhorn':False })
                        K_negative = kernel.fit_transform(Gs_negative)

                        K = np.multiply(K_plus, K_negative)
                    else:
                        kernel = WWL.WWL(param = {'discount':kernel_params['w'],'h':kernel_params['h'], 'sinkhorn':False })
                        K = kernel.fit_transform(Gs_abs)

                elif graph_label == "wwl_bin":

                    if do_tensor:

                        Gs_plus = prepare_gs_for_sp(Gs_plus, round_node= kernel_params['round_node'], round_edge= 2)
                        Gs_negative = prepare_gs_for_sp(Gs_negative, round_node= kernel_params['round_node'], round_edge=2)

                        kernel = WWL.WWL(param = {'discount':kernel_params['w'],'h':kernel_params['h'], 'sinkhorn':False })
                        K_plus = kernel.fit_transform(Gs_plus)

                        kernel = WWL.WWL(param = {'discount':kernel_params['w'],'h':kernel_params['h'], 'sinkhorn':False })
                        K_negative = kernel.fit_transform(Gs_negative)

                        K = np.multiply(K_plus, K_negative)
                    else:
                        Gs = prepare_gs_for_sp(Gs, round_node= kernel_params['round_node'], round_edge= 2)
                        kernel = WWL.WWL(param = {'discount':kernel_params['w'],'h':kernel_params['h'], 'sinkhorn':False })
                        K = kernel.fit_transform(Gs)

                elif graph_label == 'pyramid':
                    if do_tensor:
                        pm = gk.PyramidMatch(with_labels = kernel_params['with_labels'], L = kernel_params['L'], d = kernel_params['d'])
                        gk_gs = gk.graph_from_networkx(Gs_plus, edge_weight_tag='weight',  node_labels_tag = 'label')
                        K_plus = pm.fit_transform(gk_gs)

                        pm = gk.PyramidMatch(with_labels = kernel_params['with_labels'], L = kernel_params['L'], d = kernel_params['d'])
                        gk_gs = gk.graph_from_networkx(Gs_negative, edge_weight_tag='weight',  node_labels_tag = 'label')
                        K_negative = pm.fit_transform(gk_gs)

                        K = np.multiply(K_plus, K_negative)
                    else:
                        pm = gk.PyramidMatch(with_labels = kernel_params['with_labels'], L = kernel_params['L'], d = kernel_params['d'])
                        gk_gs = gk.graph_from_networkx(Gs, edge_weight_tag='weight',  node_labels_tag = 'label')
                        K = pm.fit_transform(gk_gs)

                elif graph_label == 'prop':
                    if do_tensor:
                        prop = PropagationAttr(w = kernel_params['w'],t_max = kernel_params['t_max'])
                        gk_gs = gk.graph_from_networkx(Gs_plus, edge_weight_tag='weight',  node_labels_tag = 'attr')
                        K_plus = prop.fit_transform(gk_gs)

                        prop = PropagationAttr(w = kernel_params['w'],t_max = kernel_params['t_max'])
                        gk_gs = gk.graph_from_networkx(Gs_negative, edge_weight_tag='weight',  node_labels_tag = 'attr')
                        K_negative = prop.fit_transform(gk_gs)

                        K = np.multiply(K_plus, K_negative)
                    else:
                        prop = PropagationAttr(w = kernel_params['w'],t_max = kernel_params['t_max'])
                        gk_gs = gk.graph_from_networkx(Gs_abs, edge_weight_tag='weight',  node_labels_tag = 'attr')
                        K = prop.fit_transform(gk_gs)

                elif graph_label == 'sp_attr':
                    if do_tensor:
                        Gs_sp_plus = prepare_gs_for_sp(Gs_plus, round_node= kernel_params['round_node'], round_edge= kernel_params['round_edge'])
                        Gs_sp_negative = prepare_gs_for_sp(Gs_negative, round_node= kernel_params['round_node'], round_edge= kernel_params['round_edge'])

                        init_kernel = gk.ShortestPath(normalize=0, with_labels=True)
                        graph_list = gk.graph_from_networkx(Gs_sp_plus, edge_weight_tag='weight', node_labels_tag= 'label')  # Convert to graphs to Grakel format
                        K_plus = init_kernel.fit_transform(graph_list)

                        init_kernel = gk.ShortestPath(normalize=0, with_labels=True)
                        graph_list = gk.graph_from_networkx(Gs_sp_negative, edge_weight_tag='weight', node_labels_tag= 'label')  # Convert to graphs to Grakel format
                        K_negative = init_kernel.fit_transform(graph_list)

                        K = np.multiply(K_plus, K_negative)
                    else:
                        Gs_abs_sp = prepare_gs_for_sp(Gs_abs, round_node= kernel_params['round_node'], round_edge= kernel_params['round_edge'])
                        init_kernel = gk.ShortestPath(normalize=0, with_labels=True)
                        graph_list = gk.graph_from_networkx(Gs_abs_sp, edge_weight_tag='weight', node_labels_tag= 'label')  # Convert to graphs to Grakel format
                        K = init_kernel.fit_transform(graph_list)

                elif graph_label == 'sp':
                    if do_tensor:
                        Gs_sp_plus = prepare_gs_for_sp(Gs_plus, round_node= kernel_params['round_node'], round_edge= kernel_params['round_edge'])
                        Gs_sp_negative = prepare_gs_for_sp(Gs_negative, round_node= kernel_params['round_node'], round_edge= kernel_params['round_edge'])

                        init_kernel = gk.ShortestPath(normalize=0, with_labels=False)
                        graph_list = gk.graph_from_networkx(Gs_sp_plus, edge_weight_tag='weight')  # Convert to graphs to Grakel format
                        K_plus = init_kernel.fit_transform(graph_list)

                        init_kernel = gk.ShortestPath(normalize=0, with_labels=False)
                        graph_list = gk.graph_from_networkx(Gs_sp_negative, edge_weight_tag='weight')  # Convert to graphs to Grakel format
                        K_negative = init_kernel.fit_transform(graph_list)

                        K = np.multiply(K_plus, K_negative)
                    else:
                        Gs_abs_sp = prepare_gs_for_sp(Gs_abs, round_node= kernel_params['round_node'], round_edge= kernel_params['round_edge'])
                        init_kernel = gk.ShortestPath(normalize=0, with_labels=False)
                        graph_list = gk.graph_from_networkx(Gs_abs_sp, edge_weight_tag='weight')  # Convert to graphs to Grakel format
                        K = init_kernel.fit_transform(graph_list)


                else:
                    ValueError(f"Check if graph_label is written correctly")

                MMD_functions = [mg.MMD_b, mg.MMD_u, mg.MMD_l, mg.MONK_EST]
                kernel_hypothesis = mg.BoostrapMethods(MMD_functions)
                function_arguments = [dict(n1 = n, n2 = m ), 
                                    dict(n1 = n, n2 = m ),
                                    dict(n1 = n, n2 = m ), 
                                    dict(Q = 5, n1 = n, n2 = m )]
                kernel_hypothesis.Bootstrap(K, function_arguments, B = B)

                info_dict = dict()
                info_dict['sector'] = sector_name
                info_dict['group_i'] = group_1
                info_dict['group_j'] = group_2
                info_dict['MMD_u'] = kernel_hypothesis.p_values['MMD_u']
                info_dict['MMD_b'] = kernel_hypothesis.p_values['MMD_b']
                info_dict['MMD_l'] = kernel_hypothesis.p_values['MMD_l']
                info_dict['MONK_EST'] = kernel_hypothesis.p_values['MONK_EST']
                info_dict['kernel'] = "rw"
                info_dict['r'] = r
                info_dict['dates'] = dates[i]
                info_dict['dates_mid'] = dates[int((i+(i-n))/2)]

                for k, v in kernel_params.items():
                    info_dict[k] = v

                esg_return_df = pd.concat((esg_return_df, pd.DataFrame(info_dict, index = [0])), ignore_index=True)

                pbar.update()
    pbar.close()

    L= {'info_dict':esg_return_df, 'sector':k, 'n':n, 'dates':dates[n:], 'graph_dict':graph_name, 'transform':transform, 'scale':scale}

    path = f'data/mmd_test/{study}/n_{n}_B_{B}_dstep_{day_step}_glabel_{graph_label}_p_{ptype}_tensor_{do_tensor}'
    for k,v in kernel_params.items():
        path = path + f"_{k}" + f"_{v}"
    path = path + ".pkl"

    with open(path, 'wb') as f:
        pickle.dump(L, f)

    return None

def kernel_test_iteration(i, data_dict, study, transform, scale, n,graph_name, ptype, B, edge_attr = 'weight',day_step = 1, graph_label = None, do_tensor = False, kernel_params = None):
    start_time = time.time()
    m = n
    group_1 = 0
    group_2 = 2

    graph_dict = data_dict[graph_name]
    k = data_dict['sector']
    dates = data_dict['dates']

    
    if graph_name == 'cov_dict':
        Gs = [ nx.from_numpy_array(1e4*graph_dict[group_1][s]) for s in range(i-n, i )] + [ nx.from_numpy_array(1e4*graph_dict[group_2][s]) for s in range(i-n, i )]
    else:
        Gs = [ graph_dict[group_1][s] for s in range(i-n, i )] + [ graph_dict[group_2][s] for s in range(i-n, i )]
    

    # rw nr eigenvalues
    r = np.min((6, Gs[0].number_of_nodes()-1))

    # get attributes
    if ptype is None:
        p = None
        q = None
    elif ptype == 'return':
        if graph_label == 'rw':
            p = np.vstack(([ data_dict['return_dict'][group_1][s]*1000 for s in range(i-n, i )],[ data_dict['return_dict'][group_2][s]*1000  for s in range(i-n, i )]))
        else:
            p = np.vstack(([ data_dict['return_dict'][group_1][s] for s in range(i-n, i )],[ data_dict['return_dict'][group_2][s]  for s in range(i-n, i )]))
        q = p.copy()
    elif ptype =='sharpe_dratio':
        p = []
        for s in range(i-n, i ):
            p_tmp = data_dict['portfolios_info']['sharpe']['weights'][group_1][s]
            std = np.inner(p_tmp, np.sqrt(np.diag(data_dict['cov_dict'][group_1][s])))
            p.append( p_tmp/ std/100 )
        for s in range(i-n, i ):
            p_tmp = data_dict['portfolios_info']['sharpe']['weights'][group_2][s]
            std = np.inner(p_tmp, np.sqrt(np.diag(data_dict['cov_dict'][group_2][s])))
            p.append( p_tmp/ std/100 )
        q = p.copy()
    else:
        p = np.vstack(([ data_dict['portfolios_info'][ptype]['weights'][group_1][s] for s in range(i-n, i )],[ data_dict['portfolios_info'][ptype]['weights'][group_2][s] for s in range(i-n, i )]))
        q = p.copy()

    for s in range(len(Gs)):
        if graph_label == 'prop':
            nx.set_node_attributes(Gs[s], {j:[p[s,j]] for j in range(Gs[s].number_of_nodes())}, 'attr')
        else:
            nx.set_node_attributes(Gs[s], {j:p[s,j] for j in range(Gs[s].number_of_nodes())}, 'attr')
        nx.set_node_attributes(Gs[s], {j:str(k) for j,k in Gs[s].degree}, "label")

    if do_tensor:
        Gs_plus = []
        Gs_negative = []
        for s in range(len(Gs)):
            A = nx.adjacency_matrix(Gs[s]).todense()
            A_plus = A.copy()
            A_plus[A_plus<0] =0
            Gs_plus.append(nx.from_numpy_array(A_plus))
            nx.set_node_attributes(Gs_plus[s], nx.get_node_attributes(Gs[s], 'attr'), "attr")
            nx.set_node_attributes(Gs_plus[s], {j:str(k) for j,k in Gs[s].degree}, "label")


            A_negative = A.copy()
            A_negative[A_negative>0] =0
            A_negative = np.abs(A_negative)
            Gs_negative.append(nx.from_numpy_array(A_negative))
            nx.set_node_attributes(Gs_negative[s], nx.get_node_attributes(Gs[s], 'attr'), "attr")
            nx.set_node_attributes(Gs_negative[s], {j:str(k) for j,k in Gs[s].degree}, "label")


    if graph_label == 'rw':
        calc_ok = False
        c_new = kernel_params['c']
        while not calc_ok:
            calc_ok = True
            try:
                rw_kernel = rw.RandomWalk(Gs, c = c_new, normalize=0, p=p, q = q)
                K = rw_kernel.fit_ARKU_plus(r = r, normalize_adj=False, verbose=False, edge_attr = edge_attr)

                v,_ = np.linalg.eigh(K)
                # v[np.abs(v) < 10e-5] = 0
                if np.any(v < -10e-8):
                    raise ValueError("Not psd")
            except:
                calc_ok = False
                c_new = c_new*0.8
                print(f'{study} {graph_name} {graph_label} {ptype} {dates[i]} new c is {c_new}')
    elif graph_label == 'wl':
        if do_tensor:
            kernel = [{"name": "weisfeiler_lehman", "n_iter":kernel_params['h']}, {"name": "vertex_histogram"}]
            init_kernel = gk.GraphKernel(kernel= kernel, normalize=0)
            graph_list = gk.graph_from_networkx(Gs_plus, node_labels_tag='label')  # Convert to graphs to Grakel format
            K_plus = init_kernel.fit_transform(graph_list)

            kernel = [{"name": "weisfeiler_lehman", "n_iter":kernel_params['h']}, {"name": "vertex_histogram"}]
            init_kernel = gk.GraphKernel(kernel= kernel, normalize=0)
            graph_list = gk.graph_from_networkx(Gs_negative, node_labels_tag='label')  # Convert to graphs to Grakel format
            K_negative = init_kernel.fit_transform(graph_list)

            K = np.multiply(K_plus, K_negative)

        else:
            kernel = [{"name": "weisfeiler_lehman", "n_iter":kernel_params['h']}, {"name": "vertex_histogram"}]
            init_kernel = gk.GraphKernel(kernel= kernel, normalize=0)
            graph_list = gk.graph_from_networkx(Gs, node_labels_tag='label')  # Convert to graphs to Grakel format
            K = init_kernel.fit_transform(graph_list)

    elif graph_label == 'wloa':
        if do_tensor:
            kernel = [{"name": "WL-OA", "n_iter": kernel_params['h']}]
            init_kernel = gk.GraphKernel(kernel= kernel, normalize=0)
            graph_list = gk.graph_from_networkx(Gs_plus, node_labels_tag='label')  # Convert to graphs to Grakel format
            K_plus = init_kernel.fit_transform(graph_list)

            kernel = [{"name": "WL-OA", "n_iter": kernel_params['h']}]
            init_kernel = gk.GraphKernel(kernel= kernel, normalize=0)
            graph_list = gk.graph_from_networkx(Gs_negative, node_labels_tag='label')  # Convert to graphs to Grakel format
            K_negative = init_kernel.fit_transform(graph_list)

            K = np.multiply(K_plus, K_negative)
        else:
            kernel = [{"name": "WL-OA", "n_iter": kernel_params['h']}]
            init_kernel = gk.GraphKernel(kernel= kernel, normalize=0)
            graph_list = gk.graph_from_networkx(Gs, node_labels_tag='label')  # Convert to graphs to Grakel format
            K = init_kernel.fit_transform(graph_list)

    elif graph_label == "wwl":

        if do_tensor:
            kernel = WWL.WWL(param = {'discount':kernel_params['w'],'h':kernel_params['h'], 'sinkhorn':False })
            K_plus = kernel.fit_transform(Gs_plus)

            kernel = WWL.WWL(param = {'discount':kernel_params['w'],'h':kernel_params['h'], 'sinkhorn':False })
            K_negative = kernel.fit_transform(Gs_negative)

            K = np.multiply(K_plus, K_negative)
        else:
            kernel = WWL.WWL(param = {'discount':kernel_params['w'],'h':kernel_params['h'], 'sinkhorn':False })
            K = kernel.fit_transform(Gs)
    elif graph_label == 'pyramid':

        if do_tensor:
            pm = gk.PyramidMatch(with_labels = kernel_params['with_labels'], L = kernel_params['L'], d = kernel_params['d'])
            gk_gs = gk.graph_from_networkx(Gs_plus, edge_weight_tag='weight',  node_labels_tag = 'label')
            K_plus = pm.fit_transform(gk_gs)

            pm = gk.PyramidMatch(with_labels = kernel_params['with_labels'], L = kernel_params['L'], d = kernel_params['d'])
            gk_gs = gk.graph_from_networkx(Gs_negative, edge_weight_tag='weight',  node_labels_tag = 'label')
            K_negative = pm.fit_transform(gk_gs)

            K = np.multiply(K_plus, K_negative)
        else:
            pm = gk.PyramidMatch(with_labels = kernel_params['with_labels'], L = kernel_params['L'], d = kernel_params['d'])
            gk_gs = gk.graph_from_networkx(Gs, edge_weight_tag='weight',  node_labels_tag = 'label')
            K = pm.fit_transform(gk_gs)

    elif graph_label == 'prop':

        if do_tensor:
            prop = PropagationAttr(w = kernel_params['w'],t_max = kernel_params['t_max'])
            gk_gs = gk.graph_from_networkx(Gs_plus, edge_weight_tag='weight',  node_labels_tag = 'attr')
            K_plus = prop.fit_transform(gk_gs)

            prop = PropagationAttr(w = kernel_params['w'],t_max = kernel_params['t_max'])
            gk_gs = gk.graph_from_networkx(Gs_negative, edge_weight_tag='weight',  node_labels_tag = 'attr')
            K_negative = prop.fit_transform(gk_gs)

            K = np.multiply(K_plus, K_negative)
        else:
            prop = PropagationAttr(w = kernel_params['w'],t_max = kernel_params['t_max'])
            gk_gs = gk.graph_from_networkx(Gs, edge_weight_tag='weight',  node_labels_tag = 'attr')
            K = prop.fit_transform(gk_gs)

    elif graph_label == 'sp_attr':
        def edge_kernel(x,y):
            return np.exp(kernel_params['w_edge']*np.abs(x-y))
        def node_kernel(x,y):
            return np.exp(kernel_params['w_node']*np.abs(x-y))

        my_sp = sp.sp_kernel(weight = 'weight', with_labels = True,  edge_kernel = edge_kernel, node_kernel = node_kernel)
        if do_tensor:
            K_plus = my_sp.fit_transform(Gs_plus,verbose = False)
            K_negative = my_sp.fit_transform(Gs_negative,verbose = False)
            K = np.multiply(K_plus, K_negative)
            v,_ = np.linalg.eigh(K)
            if np.any(v < -10e-8):
                raise ValueError("Not psd sp")
        else:
            Gs_plus = []
            for i in range(len(Gs)):
                A = nx.adjacency_matrix(Gs[i]).todense()
                A_plus = np.abs(A.copy())
                Gs_plus.append(nx.from_numpy_array(A_plus))
                nx.set_node_attributes(Gs_plus[i], nx.get_node_attributes(Gs[i], 'attr'), "attr")
                nx.set_node_attributes(Gs_plus[i], {j:str(k) for j,k in Gs[i].degree}, "label")

            K = my_sp.fit_transform(Gs_plus,verbose = False)
            v,_ = np.linalg.eigh(K)
            if np.any(v < -10e-8):
                raise ValueError("Not psd sp")


    else:
        ValueError(f"Check if graph_label is written correctly")

    MMD_functions = [mg.MMD_b, mg.MMD_u, mg.MMD_l, mg.MONK_EST]
    kernel_hypothesis = mg.BoostrapMethods(MMD_functions)
    function_arguments = [dict(n = n, m = m ), 
                        dict(n = n, m = m ),
                        dict(n = n, m = m ), 
                        dict(Q = 5, y1 = list(range(n)), y2 = list(range(n,n+n)) )]
    kernel_hypothesis.Bootstrap(K, function_arguments, B = B)

    info_dict = dict()
    info_dict['sector'] = k
    info_dict['group_i'] = group_1
    info_dict['group_j'] = group_2
    info_dict['MMD_u'] = kernel_hypothesis.p_values['MMD_u']
    info_dict['MMD_b'] = kernel_hypothesis.p_values['MMD_b']
    info_dict['MMD_l'] = kernel_hypothesis.p_values['MMD_l']
    info_dict['MONK_EST'] = kernel_hypothesis.p_values['MONK_EST']
    info_dict['kernel'] = "rw"
    info_dict['r'] = r
    info_dict['dates'] = dates[i]
    info_dict['dates_mid'] = dates[int((i+(i-n))/2)]

    for k, v in kernel_params.items():
        info_dict[k] = v

    print("--- %s seconds ---" % (time.time() - start_time))
    return pd.DataFrame(info_dict, index = [0])



if __name__ == '__main__':

    d = 1
    winow_len = 300
    graph_estimation = 'huge_glasso_ebic'
    edge_attr = 'weight'

    n = 20
    day_step = 2

    B = 5000

    
    # study = 'Industrials'
    # if study == 'all':
    #     file = f'data/Graphs/Striped_{study}_d_1_winlen_300_gest_huge_glasso_ebic_scale_{False}_trans_{"nonparanormal"}.pkl'
    # else:
    #     file = f'data/Graphs/Striped_{study}_d_1_winlen_300_gest_huge_glasso_ebic_scale_{False}_trans_{"nonparanormal"}.pkl'
    # with open(file, 'rb') as f:
    #     data_dict = pickle.load(f)


    study = 'Basic Materials'
    file = f'data/Graphs/Striped_{study}_d_1_winlen_300_gest_huge_glasso_ebic_scale_{False}_trans_{"nonparanormal"}.pkl'
    with open(file, 'rb') as f:
        bm_data_dict = pickle.load(f)

    study = 'Communication Services'
    file = f'data/Graphs/Striped_{study}_d_1_winlen_300_gest_huge_glasso_ebic_scale_{False}_trans_{"nonparanormal"}.pkl'
    with open(file, 'rb') as f:
        cs_data_dict = pickle.load(f)

    study = 'Consumer Cyclical'
    file = f'data/Graphs/Striped_{study}_d_1_winlen_300_gest_huge_glasso_ebic_scale_{False}_trans_{"nonparanormal"}.pkl'
    with open(file, 'rb') as f:
        cc_data_dict = pickle.load(f)
    
    study = 'Consumer Defensive'
    file = f'data/Graphs/Striped_{study}_d_1_winlen_300_gest_huge_glasso_ebic_scale_{False}_trans_{"nonparanormal"}.pkl'
    with open(file, 'rb') as f:
        cd_data_dict = pickle.load(f)

    study = 'Real Estate'
    file = f'data/Graphs/Striped_{study}_d_1_winlen_300_gest_huge_glasso_ebic_scale_{False}_trans_{"nonparanormal"}.pkl'
    with open(file, 'rb') as f:
        re_data_dict = pickle.load(f)

    study = 'Healthcare'
    file = f'data/Graphs/Striped_{study}_d_1_winlen_300_gest_huge_glasso_ebic_scale_{False}_trans_{"nonparanormal"}.pkl'
    with open(file, 'rb') as f:
        health_data_dict = pickle.load(f)

    study = 'Energy'
    file = f'data/Graphs/Striped_{study}_d_1_winlen_300_gest_huge_glasso_ebic_scale_{False}_trans_{"nonparanormal"}.pkl'
    with open(file, 'rb') as f:
        energy_data_dict = pickle.load(f)

    study = 'Technology'
    file = f'data/Graphs/Striped_{study}_d_1_winlen_300_gest_huge_glasso_ebic_scale_{False}_trans_{"nonparanormal"}.pkl'
    with open(file, 'rb') as f:
        tech_data_dict = pickle.load(f)

    study = 'Industrials'
    file = f'data/Graphs/Striped_{study}_d_1_winlen_300_gest_huge_glasso_ebic_scale_{False}_trans_{"nonparanormal"}.pkl'
    with open(file, 'rb') as f:
        ind_data_dict = pickle.load(f)

    study = 'Financial Services'
    file = f'data/Graphs/Striped_{study}_d_1_winlen_300_gest_huge_glasso_ebic_scale_{False}_trans_{"nonparanormal"}.pkl'
    with open(file, 'rb') as f:
        fs_data_dict = pickle.load(f)


    with Pool(9) as pool:
        L = pool.starmap(graph_test, [( data_dict, study, transform, scale,
                                        n, 
                                        gtype,  
                                        ptype,
                                        B,
                                        edge_attr,
                                        day_step,
                                        graph_label,
                                        do_tensor,
                                        kernel_params) for  data_dict, study, transform, scale, ptype, gtype, graph_label, kernel_params, do_tensor in 
                                        [

                                        

                                         (fs_data_dict,'Financial Services','nonparanormal', False, 'return', 'graph_dict', 'sp_attr', {'round_node':1, 'round_edge':2}, False),
                                         (ind_data_dict,'Industrials','nonparanormal', False, 'return', 'graph_dict', 'sp_attr', {'round_node':1, 'round_edge':2}, False),
                                         

                                         (fs_data_dict,'Financial Services','nonparanormal', False, 'return', 'graph_dict', 'wl', {'h':2}, False),
                                         (ind_data_dict,'Industrials','nonparanormal', False, 'return', 'graph_dict', 'wl', {'h':2}, False),


                                         (fs_data_dict,'Financial Services','nonparanormal', False, 'return', 'graph_dict', 'wloa', {'h':2}, False),
                                         (ind_data_dict,'Industrials','nonparanormal', False, 'return', 'graph_dict', 'wloa', {'h':2}, False),


                                         (fs_data_dict,'Financial Services','nonparanormal', False, 'return', 'graph_dict', 'wwl', {'w':0.1, 'h':2}, False),
                                         (ind_data_dict,'Industrials','nonparanormal', False, 'return', 'graph_dict', 'wwl', {'w':0.1, 'h':2}, False),


                                         (fs_data_dict,'Financial Services','nonparanormal', False, 'return', 'graph_dict', 'pyramid', {'L':8, 'd':6, 'with_labels':False}, False),
                                         (ind_data_dict,'Industrials','nonparanormal', False, 'return', 'graph_dict', 'pyramid', {'L':8, 'd':6, 'with_labels':False}, False),


                                         (fs_data_dict,'Financial Services','nonparanormal', False, 'return', 'graph_dict', 'pyramid', {'L':8, 'd':6, 'with_labels':True}, False),
                                         (ind_data_dict,'Industrials','nonparanormal', False, 'return', 'graph_dict', 'pyramid', {'L':8, 'd':6, 'with_labels':True}, False),

                                         (fs_data_dict,'Financial Services','nonparanormal', False, 'uniform', 'graph_dict', 'rw', {'c':1e-5}, False),
                                         (ind_data_dict,'Industrials','nonparanormal', False, 'uniform', 'graph_dict', 'rw', {'c':1e-5}, False),

                                         (fs_data_dict,'Financial Services','nonparanormal', False, 'return', 'graph_dict', 'rw', {'c':1e-5}, False),
                                         (ind_data_dict,'Industrials','nonparanormal', False, 'return', 'graph_dict', 'rw', {'c':1e-5}, False),

                                         (fs_data_dict,'Financial Services','nonparanormal', False, 'return', 'graph_dict', 'sp', {'round_node':1, 'round_edge':2}, False),
                                         (ind_data_dict,'Industrials','nonparanormal', False, 'return', 'graph_dict', 'sp', {'round_node':1, 'round_edge':2}, False)
                                         
                                         ]])
    # with Pool(10) as pool:
    #     L = pool.starmap(graph_test, [( data_dict, study, transform, scale,
    #                                     n, 
    #                                     gtype,  
    #                                     ptype,
    #                                     B,
    #                                     edge_attr,
    #                                     day_step,
    #                                     graph_label,
    #                                     do_tensor,
    #                                     kernel_params) for  data_dict, study, transform, scale, ptype, gtype, graph_label, kernel_params, do_tensor in 
    #                                     [
    #                                      (bm_data_dict,'Basic Materials','nonparanormal', False, 'return', 'graph_dict', 'sp_attr', {'round_node':1, 'round_edge':2}, False),
    #                                      (cs_data_dict,'Communication Services','nonparanormal', False, 'return', 'graph_dict', 'sp_attr', {'round_node':1, 'round_edge':2}, False),
    #                                      (cc_data_dict,'Consumer Cyclical','nonparanormal', False, 'return', 'graph_dict', 'sp_attr', {'round_node':1, 'round_edge':2}, False),
    #                                      (cd_data_dict,'Consumer Defensive','nonparanormal', False, 'return', 'graph_dict', 'sp_attr', {'round_node':1, 'round_edge':2}, False),
    #                                      (re_data_dict,'Real Estate','nonparanormal', False, 'return', 'graph_dict', 'sp_attr', {'round_node':1, 'round_edge':2}, False),
    #                                      (health_data_dict,'Healthcare','nonparanormal', False, 'return', 'graph_dict', 'sp_attr', {'round_node':1, 'round_edge':2}, False),
    #                                      (energy_data_dict,'Energy','nonparanormal', False, 'return', 'graph_dict', 'sp_attr', {'round_node':1, 'round_edge':2}, False),
    #                                      (tech_data_dict,'Technology','nonparanormal', False, 'return', 'graph_dict', 'sp_attr', {'round_node':1, 'round_edge':2}, False),
    #                                      (fs_data_dict,'Financial Services','nonparanormal', False, 'return', 'graph_dict', 'sp_attr', {'round_node':1, 'round_edge':2}, False),
    #                                      (ind_data_dict,'Industrial','nonparanormal', False, 'return', 'graph_dict', 'sp_attr', {'round_node':1, 'round_edge':2}, False),
                                         
    #                                      (bm_data_dict,'Basic Materials','nonparanormal', False, 'return', 'graph_dict', 'wl', { 'h':2}, False),
    #                                      (cs_data_dict,'Communication Services','nonparanormal', False, 'return', 'graph_dict', 'wl', {'h':2}, False),
    #                                      (cc_data_dict,'Consumer Cyclical','nonparanormal', False, 'return', 'graph_dict', 'wl', { 'h':2}, False),
    #                                      (cd_data_dict,'Consumer Defensive','nonparanormal', False, 'return', 'graph_dict', 'wl', { 'h':2}, False),
    #                                      (re_data_dict,'Real Estate','nonparanormal', False, 'return', 'graph_dict', 'wl', { 'h':2}, False),
    #                                      (health_data_dict,'Healthcare','nonparanormal', False, 'return', 'graph_dict', 'wl', { 'h':2}, False),
    #                                      (energy_data_dict,'Energy','nonparanormal', False, 'return', 'graph_dict', 'wl', { 'h':2}, False),
    #                                      (tech_data_dict,'Technology','nonparanormal', False, 'return', 'graph_dict', 'wl', {'h':2}, False),
    #                                      (fs_data_dict,'Financial Services','nonparanormal', False, 'return', 'graph_dict', 'wl', {'h':2}, False),
    #                                      (ind_data_dict,'Industrial','nonparanormal', False, 'return', 'graph_dict', 'wl', {'h':2}, False),

    #                                      (bm_data_dict,'Basic Materials','nonparanormal', False, 'return', 'graph_dict', 'wloa', { 'h':2}, False),
    #                                      (cs_data_dict,'Communication Services','nonparanormal', False, 'return', 'graph_dict', 'wloa', {'h':2}, False),
    #                                      (cc_data_dict,'Consumer Cyclical','nonparanormal', False, 'return', 'graph_dict', 'wloa', { 'h':2}, False),
    #                                      (cd_data_dict,'Consumer Defensive','nonparanormal', False, 'return', 'graph_dict', 'wloa', { 'h':2}, False),
    #                                      (re_data_dict,'Real Estate','nonparanormal', False, 'return', 'graph_dict', 'wloa', { 'h':2}, False),
    #                                      (health_data_dict,'Healthcare','nonparanormal', False, 'return', 'graph_dict', 'wloa', { 'h':2}, False),
    #                                      (energy_data_dict,'Energy','nonparanormal', False, 'return', 'graph_dict', 'wloa', { 'h':2}, False),
    #                                      (tech_data_dict,'Technology','nonparanormal', False, 'return', 'graph_dict', 'wloa', {'h':2}, False),
    #                                      (fs_data_dict,'Financial Services','nonparanormal', False, 'return', 'graph_dict', 'wloa', {'h':2}, False),
    #                                      (ind_data_dict,'Industrial','nonparanormal', False, 'return', 'graph_dict', 'wloa', {'h':2}, False),


    #                                      (bm_data_dict,'Basic Materials','nonparanormal', False, 'return', 'graph_dict', 'wwl', {'w':0.1, 'h':2}, False),
    #                                      (cs_data_dict,'Communication Services','nonparanormal', False, 'return', 'graph_dict', 'wwl', {'w':0.1, 'h':2}, False),
    #                                      (cc_data_dict,'Consumer Cyclical','nonparanormal', False, 'return', 'graph_dict', 'wwl', {'w':0.1, 'h':2}, False),
    #                                      (cd_data_dict,'Consumer Defensive','nonparanormal', False, 'return', 'graph_dict', 'wwl', {'w':0.1, 'h':2}, False),
    #                                      (re_data_dict,'Real Estate','nonparanormal', False, 'return', 'graph_dict', 'wwl', {'w':0.1, 'h':2}, False),
    #                                      (health_data_dict,'Healthcare','nonparanormal', False, 'return', 'graph_dict', 'wwl', {'w':0.1, 'h':2}, False),
    #                                      (energy_data_dict,'Energy','nonparanormal', False, 'return', 'graph_dict', 'wwl', {'w':0.1, 'h':2}, False),
    #                                      (tech_data_dict,'Technology','nonparanormal', False, 'return', 'graph_dict', 'wwl', {'w':0.1, 'h':2}, False),
    #                                      (fs_data_dict,'Financial Services','nonparanormal', False, 'return', 'graph_dict', 'wwl', {'w':0.1, 'h':2}, False),
    #                                      (ind_data_dict,'Industrial','nonparanormal', False, 'return', 'graph_dict', 'wwl', {'w':0.1, 'h':2}, False),

    #                                      (bm_data_dict,'Basic Materials','nonparanormal', False, 'return', 'graph_dict', 'pyramid', {'L':8, 'd':6, 'with_labels':False}, False),
    #                                      (cs_data_dict,'Communication Services','nonparanormal', False, 'return', 'graph_dict', 'pyramid', {'L':8, 'd':6, 'with_labels':False}, False),
    #                                      (cc_data_dict,'Consumer Cyclical','nonparanormal', False, 'return', 'graph_dict', 'pyramid', {'L':8, 'd':6, 'with_labels':False}, False),
    #                                      (cd_data_dict,'Consumer Defensive','nonparanormal', False, 'return', 'graph_dict', 'pyramid',{'L':8, 'd':6, 'with_labels':False}, False),
    #                                      (re_data_dict,'Real Estate','nonparanormal', False, 'return', 'graph_dict', 'pyramid', {'L':8, 'd':6, 'with_labels':False}, False),
    #                                      (health_data_dict,'Healthcare','nonparanormal', False, 'return', 'graph_dict', 'pyramid', {'L':8, 'd':6, 'with_labels':False}, False),
    #                                      (energy_data_dict,'Energy','nonparanormal', False, 'return', 'graph_dict', 'pyramid', {'L':8, 'd':6, 'with_labels':False}, False),
    #                                      (tech_data_dict,'Technology','nonparanormal', False, 'return', 'graph_dict', 'pyramid', {'L':8, 'd':6, 'with_labels':False}, False),
    #                                      (fs_data_dict,'Financial Services','nonparanormal', False, 'return', 'graph_dict', 'pyramid', {'L':8, 'd':6, 'with_labels':False}, False),
    #                                      (ind_data_dict,'Industrial','nonparanormal', False, 'return', 'graph_dict', 'pyramid', {'L':8, 'd':6, 'with_labels':False}, False),

    #                                      (bm_data_dict,'Basic Materials','nonparanormal', False, 'return', 'graph_dict', 'pyramid', {'L':8, 'd':6, 'with_labels':True}, False),
    #                                      (cs_data_dict,'Communication Services','nonparanormal', False, 'return', 'graph_dict', 'pyramid', {'L':8, 'd':6, 'with_labels':True}, False),
    #                                      (cc_data_dict,'Consumer Cyclical','nonparanormal', False, 'return', 'graph_dict', 'pyramid', {'L':8, 'd':6, 'with_labels':True}, False),
    #                                      (cd_data_dict,'Consumer Defensive','nonparanormal', False, 'return', 'graph_dict', 'pyramid',{'L':8, 'd':6, 'with_labels':True}, False),
    #                                      (re_data_dict,'Real Estate','nonparanormal', False, 'return', 'graph_dict', 'pyramid', {'L':8, 'd':6, 'with_labels':True}, False),
    #                                      (health_data_dict,'Healthcare','nonparanormal', False, 'return', 'graph_dict', 'pyramid', {'L':8, 'd':6, 'with_labels':True}, False),
    #                                      (energy_data_dict,'Energy','nonparanormal', False, 'return', 'graph_dict', 'pyramid', {'L':8, 'd':6, 'with_labels':True}, False),
    #                                      (tech_data_dict,'Technology','nonparanormal', False, 'return', 'graph_dict', 'pyramid', {'L':8, 'd':6, 'with_labels':True}, False),
    #                                      (fs_data_dict,'Financial Services','nonparanormal', False, 'return', 'graph_dict', 'pyramid', {'L':8, 'd':6, 'with_labels':True}, False),
    #                                      (ind_data_dict,'Industrial','nonparanormal', False, 'return', 'graph_dict', 'pyramid', {'L':8, 'd':6, 'with_labels':True}, False),

    #                                      ]])



# ('nonparanormal', False, 'uniform', 'graph_dict', 'rw', {'c':10e-6}, False) ,
#                                         ('nonparanormal', False, 'returns', 'graph_dict', 'rw', {'c':10e-8}, False) ,
#                                            ('nonparanormal', False, 'return', 'graph_dict', 'prop', {'w':0.0001, 't_max':6}, False),
#                                            ('nonparanormal', False, 'return', 'graph_dict', 'prop', {'w':0.0001, 't_max':6}, True),
#                                          ('nonparanormal', False, 'return', 'graph_dict', 'sp', {'round_node':1, 'round_edge':2}, True),
#                                          ('nonparanormal', False, 'return', 'graph_dict', 'sp', {'round_node':1, 'round_edge':2}, False),
#                                          ('nonparanormal', False, 'return', 'graph_dict', 'sp_attr', {'round_node':1, 'round_edge':2}, True),
#                                          ('nonparanormal', False, 'return', 'graph_dict', 'sp_attr', {'round_node':1, 'round_edge':2}, False),
#                                          ('nonparanormal', False, 'return', 'graph_dict', 'wloa_bin', {'round_node':1, 'round_edge':2, 'h':2}, True),
#                                          ('nonparanormal', False, 'return', 'graph_dict', 'wloa_bin', {'round_node':1, 'round_edge':2, 'h':2}, False),
#                                          ('nonparanormal', False, 'return', 'graph_dict', 'wloa', {'round_node':1, 'round_edge':2, 'h':2}, True),
#                                          ('nonparanormal', False, 'return', 'graph_dict', 'wloa', {'round_node':1, 'round_edge':2, 'h':2}, False),
#                                          ('nonparanormal', False, 'return', 'graph_dict', 'pyramid', {'L':8, 'd':6, 'with_labels':False}, False),
#                                          ('nonparanormal', False, 'return', 'graph_dict', 'pyramid', {'L':8, 'd':6, 'with_labels':True}, False)

    # kernel_params = {'w_edge':-0.1, 'w_node':-1000}
    # with Pool(6) as pool:
    #     L = pool.starmap(kernel_test_iteration, [(i, data_dict, study, 'nonparanormal', False,
    #                                     n, 
    #                                     'graph_dict',  
    #                                     'return',
    #                                     B,
    #                                     edge_attr,
    #                                     day_step,
    #                                     'sp_attr',
    #                                     False,
    #                                     kernel_params) for  i in range(20, len(data_dict['graph_dict'][0]), 10)])

    # test_info = pd.concat(L, ignore_index= True)

    # L= {'info_dict':test_info, 'sector':study, 'n':n}


    # path = f'data/mmd_test/{study}/True_d_{1}_winlen_{300}_gest_{"huge_glasso_ebic"}_scale_{True}_trans_{"nonparanormal"}_gname_{"graph_dict"}_n_{n}_B_{B}_dstep_{day_step}_glabel_{"sp_attr"}_p_{"return"}'

    # for k,v in kernel_params.items():
    #     path = path + f"_{k}" + f"_{v}"
    # path = path + ".pkl"

    # with open(path, 'wb') as f:
    #     pickle.dump(L, f)


