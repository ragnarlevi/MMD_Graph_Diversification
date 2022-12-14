"""

fast Random walk kernels algorithm from http://www.cs.cmu.edu/~ukang/papers/fast_rwgk.pdf
"""


from modulefinder import packagePathMap
from networkx.classes.function import get_node_attributes
import numpy as np
from numpy.linalg import eigh, inv
from numpy.random import exponential
from sklearn.utils.extmath import randomized_svd
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import inv as sparse_inv
from scipy.sparse import kron as sparse_kron
import scipy
import networkx as nx
import tqdm
from scipy.sparse import kron


class RandomWalk():


    def __init__(self, X, c, normalize = False, p = None, q = None) -> None:
        """
        Parameters
        ---------------
        X: list of size N, of networkx graphs
        p: list of size size N containing initial probabilities for each graph. If None then a uniform probabilites are used.
        p: list of size size N containing stopping probabilities for each graph. If None then a uniform probabilites are used.
        c: scalar
        normalize: Should the kernel be normalized
        
        """

        self.X = X
        self.p = p
        self.q = q
        self.c = c
        self.normalize = normalize

        self.N = len(X)


    def fit(self, calc_type, r, edge_labels = None, label_list = None, k = None, mu_vec = None, normalize_adj = False , row_normalize_adj = False, label_name = 'label', verbose = True):
        """
        A wrapper for caclulating a kernel

        Parameters
        -------------------------------------
        calc_type: str,
            ARKU
            ARKU_plus
            ARKL
            exponential
            p-rw

        r: int, number of eigenvalues used in approximation
        edge_labels - array with the label vocabulary
        label_list: array with labels
        k - int, Nr. random walks
        mu_vec - array of size p, containing RW weight/discount.
        normalize_adj: bool, Should the adj matrix normalized? D^{-1/2}AD^{-1/2} where A is adj matrix and D is degree matrix.
        row_normalize_adj: bool, Should the adj matrixb be row normalized? D^{-1/2}AD^{-1/2} where A is adj matrix and D is degree matrix.
        label_name: str, what is the name of labels
        verbose: bool, print progress bar?


        Returns
        --------------------------------
        K: np.array, N x N, kernel matrix, N number of graphs
        
        """


        if calc_type == "ARKU":
            K = self.fit_ARKU(r, verbose)
        elif calc_type == "ARKU_plus":
            K = self.fit_ARKU_plus(r, normalize_adj = normalize_adj, verbose = verbose)
        elif calc_type == "ARKL":
            if label_list is None:
                raise ValueError("label list should be a list (not None)")
            K = self.fit_ARKL(r, label_list, normalize_adj = normalize_adj, row_normalize_adj = row_normalize_adj, verbose = verbose, label_name = label_name)
        elif calc_type == "exponential":
            K = self.fit_exponential(r, normalize_adj = normalize_adj, row_normalize_adj = row_normalize_adj, verbose = verbose)
        elif calc_type == 'p-rw':
            if k is None:
                raise ValueError("k must be int")
            if mu_vec is None:
                mu_vec = mu_vec = np.power(self.c ,range(k+1)) / np.array([np.math.factorial(i) for i in np.arange(k+1)])
            K = self.fit_random_walk(mu_vec, k, r, normalize_adj, row_normalize_adj, verbose)
        elif calc_type == 'ARKU_edge':
            K = self.fit_ARKU_edge(r, edge_labels, verbose)
        else:
            raise ValueError("calc type should be ARKU, ARKU_plus, ARKL, exponential, p-rw or ARKU_edge")

        return K

    

    def fit_ARKU(self, r, edge_attr = None, verbose = True):
        """
        Approximate random walk kernel for unlabeled nodes and asymmetric W where W is the (weighted) adjacency matrix.

        Parameters
        --------------------------
        r - int, number of eigenvalues used in approximation
        verbose - bool, print progress bar?
        """

        U_list = [None] * self.N  # left SVD matrix of each adj matrix
        Lamda_list = [None] * self.N  # eigenvalues of each adj matrix
        Vt_list = [None] * self.N  # right transposed SVD matrix of each adj matrix
        K = np.zeros((self.N, self.N))

        if verbose:
            pbar = tqdm.tqdm(disable=(verbose is False), total=self.N*(self.N+1)/2)
        #disable=(disp is False), total=(13 * len(_null_models) * len(mat_types))


        for i in range(self.N):
            for j in range(i,self.N):
                
                # Row normalize the adjacency matrix
                if U_list[i] is None:
                    W_row_normalize = self._row_normalized_adj(self.X[i], edge_attr = edge_attr)
                    U_list[i], Lamda_list[i], Vt_list[i] = randomized_svd(W_row_normalize.T, n_components= r)
                    if np.any(np.concatenate(Lamda_list[i])== 0.0):
                        raise ValueError("zero eigenvalue.")

                if U_list[j] is None:
                    W_row_normalize = self._row_normalized_adj(self.X[j], edge_attr = edge_attr)
                    U_list[j], Lamda_list[j], Vt_list[j] = randomized_svd(W_row_normalize.T, n_components= r)
                    if np.any(np.concatenate(Lamda_list[j])== 0.0):
                        raise ValueError("zero eigenvalue.")

                if (self.p is None) and (self.q is None):
                    p1 = np.ones((self.X[i].number_of_nodes())) / float(self.X[i].number_of_nodes())
                    p2 = np.ones((self.X[j].number_of_nodes())) / float(self.X[j].number_of_nodes())
                    q1 = np.ones((self.X[i].number_of_nodes())) / float(self.X[i].number_of_nodes())
                    q2 = np.ones((self.X[j].number_of_nodes())) / float(self.X[j].number_of_nodes())
                elif (self.p is None) and ~(self.q is None):
                    p1 = np.ones((self.X[i].number_of_nodes())) / float(self.X[i].number_of_nodes())
                    p2 = np.ones((self.X[j].number_of_nodes())) / float(self.X[j].number_of_nodes())
                    q1 = self.q[i]
                    q2 = self.q[j]
                elif ~(self.p is None) and (self.q is None):
                    q1 = np.ones((self.X[i].number_of_nodes())) / float(self.X[i].number_of_nodes())
                    q2 = np.ones((self.X[j].number_of_nodes())) / float(self.X[j].number_of_nodes())
                    p1 = self.p[i]
                    p2 = self.p[j] 
                else:
                    p1 = self.p[i]
                    p2 = self.p[j] 
                    q1 = self.q[i]
                    q2 = self.q[j]

                K[i,j] = self.ARKU(U_list[i], Lamda_list[i], Vt_list[i], U_list[j], Lamda_list[j], Vt_list[j], r, p1, p2, q1, q2)

                if verbose:
                    pbar.update()

        if verbose:
            pbar.close()

        K = np.triu(K) + np.triu(K, 1).T

        if self.normalize:
            K = self.normalize_gram_matrix(K)


        return K

    def fit_ARKU_edge(self, r, edge_labels, edge_attr = None, verbose = True, edge_label_tag = 'sign' ):
        """
        Approximate random walk kernel for edge labelled graph and asymmetric W where W is the (weighted) adjacency matrix.
        
        Parameters
        ------------------
        r - int, number of eigenvalues used in approximation
        edge_labels - array with the label vocabulary
        verbose - bool, print progress bar?

        Returns
        -------------------
        K - Kernel matrix
        """

        if r < 1:
            raise ValueError('r has to 1 or bigger')

        nr_edge_labels = len(edge_labels)

        U_list = [[None] * nr_edge_labels] * self.N  # left SVD matrix of each adj matrix
        Lamda_list = [[None] * nr_edge_labels] * self.N  # eigenvalues of each adj matrix
        K = np.zeros((self.N, self.N))

        if verbose:
            pbar = tqdm.tqdm(disable=(verbose is False), total=self.N*(self.N+1)/2)
        #disable=(disp is False), total=(13 * len(_null_models) * len(mat_types))


        for i in range(self.N):
            for j in range(i,self.N):
                
                if Lamda_list[i][0] is None:
                    all_A = self._get_label_adj(self.X[i].copy(), edge_labels,edge_label_tag, edge_attr)
                    Lamda_list[i], U_list[i] = self._eigen_decomp(all_A, r)
                    if np.any(np.concatenate(Lamda_list[i])== 0.0) :
                        raise ValueError("zero eigenvalue, probably too few edge labels for one class.")
                if Lamda_list[j][0] is None:
                    all_A = self._get_label_adj(self.X[j].copy(), edge_labels,edge_label_tag, edge_attr)
                    Lamda_list[j], U_list[j] = self._eigen_decomp(all_A, r)
                    if np.any(np.concatenate(Lamda_list[j])== 0.0):
                        raise ValueError("zero eigenvalue, probably too few edge labels for one class.")


                if (self.p is None) and (self.q is None):
                    p1 = np.ones((self.X[i].number_of_nodes())) / float(self.X[i].number_of_nodes())
                    p2 = np.ones((self.X[j].number_of_nodes())) / float(self.X[j].number_of_nodes())
                    q1 = np.ones((self.X[i].number_of_nodes())) / float(self.X[i].number_of_nodes())
                    q2 = np.ones((self.X[j].number_of_nodes())) / float(self.X[j].number_of_nodes())
                elif (self.p is None) and ~(self.q is None):
                    p1 = np.ones((self.X[i].number_of_nodes())) / float(self.X[i].number_of_nodes())
                    p2 = np.ones((self.X[j].number_of_nodes())) / float(self.X[j].number_of_nodes())
                    q1 = self.q[i]
                    q2 = self.q[j]
                elif ~(self.p is None) and (self.q is None):
                    q1 = np.ones((self.X[i].number_of_nodes())) / float(self.X[i].number_of_nodes())
                    q2 = np.ones((self.X[j].number_of_nodes())) / float(self.X[j].number_of_nodes())
                    p1 = self.p[i]
                    p2 = self.p[j] 
                else:
                    p1 = self.p[i]
                    p2 = self.p[j] 
                    q1 = self.q[i]
                    q2 = self.q[j]


                K[i,j] = self.ARKU_edge(U_list[i], Lamda_list[i], p1, q1, U_list[j], Lamda_list[j], p2, q2 )

                if verbose:
                    pbar.update()

        if verbose:
            pbar.close()

        K = np.triu(K) + np.triu(K, 1).T

        if self.normalize:
            K = self.normalize_gram_matrix(K)


        return K

    def ARKU_edge(self, u1, w1, p1, q1, u2, w2, p2, q2 ):
        """
        Fast Random walk kernel for symmetric (weight) matrices

        Parameters
        ----------------
        u1, u2 - 2d array, eigenvector matrix of Adjacency matrix of G1, G2
        w1, w2 - 1d array, eigenvalues of Adjacency matrix of G1, G2
        p1, p2 - initial probabilities
        q1, q2 - stopping probabilities
        """

        nr_labels = len(u1)

        # Create the label eigenvalue (block) diagonal matrix
        diag_inverse = [] 
        for i in range(nr_labels):
            diag_inverse.append(np.diag(np.kron(np.diag(np.reciprocal(w1[i])), np.diag(np.reciprocal(w2[i])))))
        diag_inverse =  np.concatenate(diag_inverse)
        Lamda = np.diag(np.reciprocal(diag_inverse - self.c ))

        L = []
        for i in range(nr_labels):
            L.append(np.kron(np.matmul(q1.T, u1[i]), np.matmul(q2.T, u2[i])))
        L = np.concatenate(L)

        R = [] 
        for i in range(nr_labels):
            R.append(np.kron(np.matmul(u1[i].T, p1), np.matmul(u2[i].T, p2)))
        R = np.concatenate(R)

        return np.inner(q1,p1)*np.inner(q2,p2) + self.c*np.dot(L, Lamda).dot(R)

    def fit_ARKU_plus(self, r, normalize_adj = False, edge_attr = None, verbose = True):
        """
        Approximate random walk kernel for unlabeled nodes and asymmetric W where W is the (weighted) adjacency matrix.

        Parameters
        --------------------------
        r - int, number of eigenvalues used in approximation
        verbose - bool, print progress bar?
        normalize_adj - bool, Should the adj matrix normalized? D^{-1/2}AD^{-1/2} where A is adj matrix and D is degree matrix.

        """
        all_A = [None] * self.N
        U_list = [None] * self.N  # eigenvector matrix of each adj matrix
        Lamda_list = [None] * self.N  # eigenvalues of each adj matrix

        K = np.zeros((self.N, self.N))

        if verbose:
            pbar = tqdm.tqdm(disable=(verbose is False), total=self.N*(self.N+1)/2)
        #disable=(disp is False), total=(13 * len(_null_models) * len(mat_types))


        for i in range(self.N):
            for j in range(i,self.N):

                if normalize_adj:
                    if all_A[i] is None:
                        all_A[i] = self._normalized_adj(self.X[i], edge_attr = edge_attr)
                        Lamda_list[i], U_list[i] = eigsh(all_A[i].T, k = r)
                    if np.any(np.concatenate(Lamda_list[i])== 0.0):
                        raise ValueError("zero eigenvalue.")
                    if all_A[j] is None:
                        all_A[j] = self._normalized_adj(self.X[j], edge_attr = edge_attr)
                        Lamda_list[j], U_list[j] = eigsh(all_A[j].T, k = r)
                    if np.any(np.concatenate(Lamda_list[j])== 0.0):
                        raise ValueError("zero eigenvalue.")
                else:
                    if all_A[i] is None:
                        all_A[i] = self._get_adj_matrix(self.X[i], edge_attr = edge_attr)
                        Lamda_list[i], U_list[i] = eigsh(all_A[i].T, k = r)
                    if all_A[j] is None:
                        all_A[j] = self._get_adj_matrix(self.X[j], edge_attr = edge_attr)
                        Lamda_list[j], U_list[j] = eigsh(all_A[j].T, k = r)


                if (self.p is None) and (self.q is None):
                    p1 = np.ones((self.X[i].number_of_nodes())) / float(self.X[i].number_of_nodes())
                    p2 = np.ones((self.X[j].number_of_nodes())) / float(self.X[j].number_of_nodes())
                    q1 = np.ones((self.X[i].number_of_nodes())) / float(self.X[i].number_of_nodes())
                    q2 = np.ones((self.X[j].number_of_nodes())) / float(self.X[j].number_of_nodes())
                elif (self.p is None) and ~(self.q is None):
                    p1 = np.ones((self.X[i].number_of_nodes())) / float(self.X[i].number_of_nodes())
                    p2 = np.ones((self.X[j].number_of_nodes())) / float(self.X[j].number_of_nodes())
                    q1 = self.q[i]
                    q2 = self.q[j]
                elif ~(self.p is None) and (self.q is None):
                    q1 = np.ones((self.X[i].number_of_nodes())) / float(self.X[i].number_of_nodes())
                    q2 = np.ones((self.X[j].number_of_nodes())) / float(self.X[j].number_of_nodes())
                    p1 = self.p[i]
                    p2 = self.p[j] 
                else:
                    p1 = self.p[i]
                    p2 = self.p[j] 
                    q1 = self.q[i]
                    q2 = self.q[j]

                K[i,j] = self.ARKU_plus(U_list[i], Lamda_list[i], U_list[j], Lamda_list[j], r, p1, p2, q1, q2)

                if verbose:
                    pbar.update()


        if verbose:
            pbar.close()

        K = np.triu(K) + np.triu(K, 1).T

        if self.normalize:
            K = self.normalize_gram_matrix(K)
        
        return K


    def fit_ARKL(self, r, label_list, normalize_adj = False, row_normalize_adj = False, edge_attr =None, verbose = True, label_name = 'label'):
        """
        Fit approximate label node random walk kernel.

        Parameters
        ----------------------------
        r - int, number of eigenvalues
        label_list - array with labels
        normalize_adj - bool, Should the adj matrix normalized? D^{-1/2}AD^{-1/2} where A is adj matrix and D is degree matrix.
        row_normalize_adj - bool, Should the adj matrix be row normalized? AD^{-1/2} where A is adj matrix and D is degree matrix.
        verbose - bool, print progress bar?
        label_name - str, what is the name of labels

        Returns 
        ------------------
        K - np.array, N x N, kernel matrix, N number of graphs
        
        """

        if normalize_adj and row_normalize_adj:
            raise ValueError("Can not have both row normalized and normalized adj") 

        
        all_A = [None] * self.N
        U_list = [None] * self.N  # left SVD matrix of each adj matrix
        Lamda_list = [None] * self.N  # eigenvalues of each adj matrix
        Vt_list = [None] * self.N  # Right transposed SVD matrix of each adj matrix
        K = np.zeros((self.N, self.N))

        if verbose:
            pbar = tqdm.tqdm(disable=(verbose is False), total=self.N*(self.N+1)/2)

        # get label matrix/vector of all graphs
        Ls = [None] * self.N
        for i in range(self.N):
            Ls[i] = self._get_node_label_vectors(self.X[i], label_list, label_name)

        for i in range(self.N):
            for j in range(i,self.N):


                if row_normalize_adj:
                    if all_A[i] is None:
                        all_A[i] = self._row_normalized_adj(self.X[i], edge_attr = edge_attr)
                        U_list[i], Lamda_list[i], Vt_list[i] = randomized_svd(all_A[i].T, n_components= r)
                    if all_A[j] is None:
                        all_A[j] = self._row_normalized_adj(self.X[j], edge_attr = edge_attr)
                        U_list[j], Lamda_list[j], Vt_list[j] = randomized_svd(all_A[j].T, n_components= r)
                else:
                    if all_A[i] is None:
                        all_A[i] = self._get_adj_matrix(self.X[i], edge_attr = edge_attr)
                        U_list[i], Lamda_list[i], Vt_list[i] = randomized_svd(all_A[i].T, n_components= r)
                    if all_A[j] is None:
                        all_A[j] = self._get_adj_matrix(self.X[j], edge_attr = edge_attr)
                        U_list[j], Lamda_list[j], Vt_list[j] = randomized_svd(all_A[j].T, n_components= r)
                

                if (self.p is None) and (self.q is None):
                    p1 = np.ones((self.X[i].number_of_nodes())) / float(self.X[i].number_of_nodes())
                    p2 = np.ones((self.X[j].number_of_nodes())) / float(self.X[j].number_of_nodes())
                    q1 = np.ones((self.X[i].number_of_nodes())) / float(self.X[i].number_of_nodes())
                    q2 = np.ones((self.X[j].number_of_nodes())) / float(self.X[j].number_of_nodes())
                elif (self.p is None) and ~(self.q is None):
                    p1 = np.ones((self.X[i].number_of_nodes())) / float(self.X[i].number_of_nodes())
                    p2 = np.ones((self.X[j].number_of_nodes())) / float(self.X[j].number_of_nodes())
                    q1 = self.q[i]
                    q2 = self.q[j]
                elif ~(self.p is None) and (self.q is None):
                    q1 = np.ones((self.X[i].number_of_nodes())) / float(self.X[i].number_of_nodes())
                    q2 = np.ones((self.X[j].number_of_nodes())) / float(self.X[j].number_of_nodes())
                    p1 = self.p[i]
                    p2 = self.p[j] 
                else:
                    p1 = self.p[i]
                    p2 = self.p[j] 
                    q1 = self.q[i]
                    q2 = self.q[j]

                K[i,j] = self.ARKL(U_list[i], Lamda_list[i], Vt_list[i], U_list[j], Lamda_list[j], Vt_list[j], r, Ls[i], Ls[j], p1, p2, q1, q2)

                if verbose:
                    pbar.update()

        if verbose:
            pbar.close()
            
        K = np.triu(K) + np.triu(K, 1).T

        if self.normalize:
            K = self.normalize_gram_matrix(K)

        return K

    def fit_exponential(self, r = None, normalize_adj = False, row_normalize_adj = False, edge_attr = None, verbose = True):
        """
        Perform an infnite exponential random walk. Does not work for labelled graphs

        Parameters
        ---------------------------------------
        r - int, number of eigenvalues, if None full eigenvalue decomposition used which will be slow.
        normalize_adj - bool, Should the adj matrix normalized? D^{-1/2}AD^{-1/2} where A is adj matrix and D is degree matrix.
        row_normalize_adj - bool, Should the adj matrixb be row normalized? D^{-1/2}AD^{-1/2} where A is adj matrix and D is degree matrix.
        verbose - bool, print progress bar?
        
        Returns
        --------------------------------
        K - N n N kernel matrix 

        """

        if normalize_adj and row_normalize_adj:
            raise ValueError("Can not have both row normalized and normalized adj") 

        
        all_A = [None] * self.N
        U_list = [None] * self.N  # eigenvector matrix of each adj matrix
        Lamda_list = [None] * self.N  # eigenvalues of each adj matrix
        K = np.zeros((self.N, self.N))

        if verbose:
            pbar = tqdm.tqdm(disable=(verbose is False), total=self.N*(self.N+1)/2)

        # normalize type, eigenvalue decomposition 
        for i in range(self.N):
            for j in range(i,self.N):
                
                if normalize_adj:
                    if all_A[i] is None:
                        all_A[i] = self._normalized_adj(self.X[i], edge_attr = edge_attr)
                        if r is None:
                            Lamda_list[i], U_list[i] = eigh(np.array(all_A[i].T.todense()))
                        else:
                            Lamda_list[i], U_list[i] = eigsh(all_A[i].T, k = r)
                    if all_A[j] is None:
                        all_A[j] = self._normalized_adj(self.X[j], edge_attr = edge_attr)
                        if r is None:
                            Lamda_list[j], U_list[j] = eigh(np.array(all_A[j].T.todense()))
                        else:
                            Lamda_list[j], U_list[j] = eigsh(all_A[j].T, k = r)
                elif row_normalize_adj:
                    if all_A[i] is None:
                        all_A[i] = self._row_normalized_adj(self.X[i], edge_attr = edge_attr)
                        if r is None:
                            Lamda_list[i], U_list[i] = eigh(np.array(all_A[i].T.todense()))
                        else:
                            Lamda_list[i], U_list[i] = eigsh(all_A[i].T, k = r)
                    if all_A[j] is None:
                        all_A[j] = self._row_normalized_adj(self.X[j], edge_attr = edge_attr)
                        if r is None:
                            Lamda_list[j], U_list[j] = eigh(np.array(all_A[j].T.todense()))
                        else:
                            Lamda_list[j], U_list[j] = eigsh(all_A[j].T, k = r)
                else:
                    if all_A[i] is None:
                        all_A[i] = self._get_adj_matrix(self.X[i], edge_attr = edge_attr)
                        if r is None:
                            Lamda_list[i], U_list[i] = eigh(np.array(all_A[i].T.todense()))
                        else:
                            Lamda_list[i], U_list[i] = eigsh(all_A[i].T, k = r)
                    if all_A[j] is None:
                        all_A[j] = self._get_adj_matrix(self.X[j], edge_attr = edge_attr)
                        if r is None:
                            Lamda_list[j], U_list[j] = eigh(np.array(all_A[j].T.todense()))
                        else:
                            Lamda_list[j], U_list[j] = eigsh(all_A[j].T, k = r)


                

                if (self.p is None) and (self.q is None):
                    p1 = np.ones((self.X[i].number_of_nodes())) / float(self.X[i].number_of_nodes())
                    p2 = np.ones((self.X[j].number_of_nodes())) / float(self.X[j].number_of_nodes())
                    q1 = np.ones((self.X[i].number_of_nodes())) / float(self.X[i].number_of_nodes())
                    q2 = np.ones((self.X[j].number_of_nodes())) / float(self.X[j].number_of_nodes())
                elif (self.p is None) and ~(self.q is None):
                    p1 = np.ones((self.X[i].number_of_nodes())) / float(self.X[i].number_of_nodes())
                    p2 = np.ones((self.X[j].number_of_nodes())) / float(self.X[j].number_of_nodes())
                    q1 = self.q[i]
                    q2 = self.q[j]
                elif ~(self.p is None) and (self.q is None):
                    q1 = np.ones((self.X[i].number_of_nodes())) / float(self.X[i].number_of_nodes())
                    q2 = np.ones((self.X[j].number_of_nodes())) / float(self.X[j].number_of_nodes())
                    p1 = self.p[i]
                    p2 = self.p[j] 
                else:
                    p1 = self.p[i]
                    p2 = self.p[j] 
                    q1 = self.q[i]
                    q2 = self.q[j]

                if r is None:
                    p1 = np.expand_dims(p1, axis = 1)
                    p2 = np.expand_dims(p2, axis = 1)
                    q1 = np.expand_dims(q1, axis = 1)
                    q2 = np.expand_dims(q2, axis = 1)


                K[i,j] = self.rw_exponential(U_list[i], Lamda_list[i], U_list[j], Lamda_list[j], p1, p2, q1, q2, r)

                if verbose:
                    pbar.update()

        if verbose:
            pbar.close()

        K = np.triu(K) + np.triu(K, 1).T

        if self.normalize:
            K = self.normalize_gram_matrix(K)

        return K

    def rw_exponential(self, u1, w1, u2, w2, p1, p2, q1, q2, r):
        """
        Perform exponential random walk. Adjacency matrix has to be symmetric.

        Parameters
        ---------------------------------------
        u1, u2 - 2d array, eigenvector matrix of Adjacency matrix of G1, G2
        w1, w2 - 1d array, eigenvalues of Adjacency matrix of G1, G2
        p1, p2 - initial probabilities
        q1, q2 - stopping probabilities
        r - int, If passed then an approximation is used by eigen decomposition
        
        Returns
        --------------------------------
        float kernel value between W1,W2

        """

        if (r is not None):
            if r < 1:
                raise ValueError('r has to 1 or bigger')

        w = np.array(np.concatenate([w*w2 for w in w1]))  # kron product

        stop_part = np.kron(np.matmul(q1.T, u1), np.matmul(q2.T, u2))
        start_part = np.kron(np.matmul(u1.T, p1), np.matmul(u2.T, p2))
        return np.matmul(np.matmul(stop_part, np.diag(np.exp(w))), start_part)



    def fit_random_walk(self, mu_vec, k, r , normalize_adj = False, row_normalize_adj = False, verbose = True, edge_attr = 'weight'):
        """
        Perform p-random walks. Symmetric matrix

        Parameters
        ---------------------------------------
        k - int, Nr. random walks
        mu_vec - array of size p, containing RW weight/discount.
        r - int, number of eigenvalues, if None full eigenvalue decomposition used which might be slow.
        normalize_adj - bool, Should the adj matrix normalized? D^{-1/2}AD^{-1/2} where A is adj matrix and D is degree matrix.
        row_normalize_adj - bool, Should the adj matrixb be row normalized? D^{-1/2}AD^{-1/2} where A is adj matrix and D is degree matrix.
        verbose - bool, print progress bar?
        
        Returns
        --------------------------------
        K - N n N kernel matrix 

        """

        if normalize_adj and row_normalize_adj:
            raise ValueError("Can not have both row normalized and normalized adj") 

        
        # all_A = [None] * self.N
        # U_list = [None] * self.N  # eigenvector matrix of each adj matrix
        # Lamda_list = [None] * self.N  # eigenvalues of each adj matrix
        K = np.zeros((self.N, self.N))

        if verbose:
            pbar = tqdm.tqdm(disable=(verbose is False), total=self.N*(self.N+1)/2)

        # normalize type, eigenvalue decomposition 
        for i in range(self.N):
            for j in range(i,self.N):

        #         if normalize_adj:
        #             if all_A[i] is None:
        #                 all_A[i] = self._normalized_adj(self.X[i])
        #             if all_A[j] is None:
        #                 all_A[j] = self._normalized_adj(self.X[j])
        #         elif row_normalize_adj:
        #             if all_A[i] is None:
        #                 all_A[i] = self._row_normalized_adj(self.X[i])
        #             if all_A[j] is None:
        #                 all_A[j] = self._row_normalized_adj(self.X[j])
        #         else:
        #             if all_A[i] is None:
        #                 all_A[i] = self._get_adj_matrix(self.X[i])
        #             if all_A[j] is None:
        #                 all_A[j] = self._get_adj_matrix(self.X[j])


                

                if (self.p is None) and (self.q is None):
                    p1 = np.ones((self.X[i].number_of_nodes())) / float(self.X[i].number_of_nodes())
                    p2 = np.ones((self.X[j].number_of_nodes())) / float(self.X[j].number_of_nodes())
                    q1 = np.ones((self.X[i].number_of_nodes())) / float(self.X[i].number_of_nodes())
                    q2 = np.ones((self.X[j].number_of_nodes())) / float(self.X[j].number_of_nodes())
                elif (self.p is None) and ~(self.q is None):
                    p1 = np.ones((self.X[i].number_of_nodes())) / float(self.X[i].number_of_nodes())
                    p2 = np.ones((self.X[j].number_of_nodes())) / float(self.X[j].number_of_nodes())
                    q1 = self.q[i]
                    q2 = self.q[j]
                elif ~(self.p is None) and (self.q is None):
                    q1 = np.ones((self.X[i].number_of_nodes())) / float(self.X[i].number_of_nodes())
                    q2 = np.ones((self.X[j].number_of_nodes())) / float(self.X[j].number_of_nodes())
                    p1 = self.p[i]
                    p2 = self.p[j] 
                else:
                    p1 = self.p[i]
                    p2 = self.p[j] 
                    q1 = self.q[i]
                    q2 = self.q[j]

        #         if r is None:
        #             p1 = np.expand_dims(p1, axis = 1)
        #             p2 = np.expand_dims(p2, axis = 1)
        #             q1 = np.expand_dims(q1, axis = 1)
        #             q2 = np.expand_dims(q2, axis = 1)
# nx.adjacency_matrix(self.X[j], weight = 'weight')
                K[i,j] = self.p_rw_symmetric(self._get_adj_matrix(self.X[i], edge_attr = edge_attr) ,self._get_adj_matrix(self.X[j], edge_attr = edge_attr), k, mu_vec, p1, p2, q1, q2, r)

                if verbose:
                    pbar.update()

        if verbose:
            pbar.close()

        K = np.triu(K) + np.triu(K, 1).T

        if self.normalize:
            K = self.normalize_gram_matrix(K)

        return K


    def p_rw_symmetric(self,A1, A2, k, mu_vec, p1, p2, q1, q2, r = None):
        """
        Perform p-random walks. Symmetric matrix

        Parameters
        ---------------------------------------
        u1, u2 - 2d array, eigenvector matrix of Adjacency matrix of G1, G2
        w1, w2 - 1d array, eigenvalues of Adjacency matrix of G1, G2
        k - int, Nr. random walks
        mu_vec - array of size p, containing RW weight/discount.
        p1, p2 - initial probabilities
        q1, q2 - stoppoing probabilities
        r - int, If passed then an approximation is used by eigen decomposition

        Returns
        --------------------------------
        float kernel value between W1,W2

        """
        if (r is not None):
            if r < 1:
                raise ValueError('r has to 1 or bigger')


        A = np.array(sparse_kron(A1,A2).todense())
        #w, u = eigsh(A.T, k= r)

        #D = np.ones(shape=(len(w)))*mu_vec[0]
        D = np.identity(A.shape[0])*mu_vec[0]
        A_mult = np.identity(A.shape[0])
        for s in range(1,k+1):
            A_mult = np.dot(A_mult, A)
            D = D + A_mult*mu_vec[s]

        pk = np.kron(p1,p2)
        qk = np.kron(q1,q2)
        return np.dot(qk, D).dot(pk)


        # for i in range(1,k+1):
        #     D = D + np.power(w,i)*mu_vec[i]

        # p = np.kron(p1,p2)
        # q = np.kron(q1,q2)
        # stop_part = np.matmul(u.T, q)
        # start_part = np.matmul(u.T, p)
        # return np.matmul(np.matmul(stop_part.T, np.diag(D)), start_part)
        



    def ARKU(self, u1, w1, v1t, u2, w2, v2t, r, p1, p2, q1, q2):
        """
        Calculate Kernel value between G1 and G2

        Parameters
        ----------------
        u1, u2 - 2d array, Left SVD matrix of each adjacency matrix of G1, G2
        w1, w2 - 1d array, Eigenvalues of each adjacency matrix of G1, G2
        v1t, v2t - 2d array, Right transposed SVD matrix of each adjacency matrix of G1, G2
        r - int how many eigenvalues?
        p1, p2 - initial probabilities
        q1, q2 - stopping probabilities
        """

        if r < 1:
            raise ValueError('r has to 1 or bigger')


        diag_inverse =  np.kron(np.diag(np.reciprocal(w1)), np.diag(np.reciprocal(w2)))
        Lamda = inv(diag_inverse - self.c * np.matmul(np.kron(v1t, v2t), np.kron(u1, u2)))
        L = np.kron(np.matmul(q1.T, u1), np.matmul(q2.T, u2))
        R = np.kron(np.matmul(v1t, p1), np.matmul(v2t, p2))

        return np.inner(q1,p1)*np.inner(q2,p2) + self.c*np.dot(L, Lamda).dot(R)


    def ARKU_plus(self, u1, w1, u2, w2, r, p1, p2, q1, q2):
        """
        Fast Random walk kernel for symmetric (weight) matrices

        Parameters
        ----------------
        u1, u2 - 2d array, eigenvector matrix of Adjacency matrix of G1, G2
        w1, w2 - 1d array, eigenvalues of Adjacency matrix of G1, G2
        r - int how many eigenvalues?
        p1, p2 - initial probabilities
        q1, q2 - stoppoing probabilities
        """

        if r < 1:
            raise ValueError('r has to 1 or bigger')

        u1 = u1[:,np.where(w1!= 0)[0]]
        w1 = w1[np.where(w1!= 0)[0]]

        u2 = u2[:,np.where(w2!= 0)[0]]
        w2 = w2[np.where(w2!= 0)[0]]
 
        diag_inverse =  np.kron(np.diag(np.reciprocal(w1)), np.diag(np.reciprocal(w2)))
        Lamda = inv(diag_inverse - self.c * np.identity(diag_inverse.shape[0]))
        L = np.kron(np.matmul(q1.T, u1), np.matmul(q2.T, u2))
        R = np.kron(np.matmul(u1.T, p1), np.matmul(u2.T, p2))

        return np.inner(q1,p1)*np.inner(q2,p2) + self.c*np.dot(L, Lamda).dot(R)

    def ARKL(self, u1, w1, v1t, u2, w2, v2t, r, L1, L2, p1, p2, q1, q2):
        """
        Fit an approximation to node labeled graphs

        Parameters
        -------------------------
        u1, u2 - 2d array, Left SVD matrix of each adjacency matrix of G1, G2
        w1, w2 - 1d array, Eigenvalues of each adjacency matrix of G1, G2
        v1t, v2t - 2d array, Right transposed SVD matrix of each adjacency matrix of G1, G2
        r - int how many eigenvalues?
        L1, L2 - list containing vectors. Each vector corresponds to a label and a element nr i is 1 if node i has the label 0 otherwise
        p1, p2 - initial probabilities
        q1, q2 - stoppoing probabilities
        """

        if r < 1:
            raise ValueError('r has to 1 or bigger')


        diag_inverse =  np.kron(np.diag(np.reciprocal(w1)), np.diag(np.reciprocal(w2)))
        Lamda = inv(diag_inverse - self.c * np.kron(np.matmul(np.matmul(v1t, np.diag(np.sum(L1, axis=0))), u1), np.matmul(np.matmul(v2t, np.diag(np.sum(L2, axis=0))), u2)))
        L = np.sum([np.kron(np.matmul(np.matmul(q1.T, np.diag(L1[i])), u1), np.matmul(np.matmul(q2.T, np.diag(L2[i])), u2)) for i in range(len(L1))], axis=0)
        R = np.sum([np.kron(np.matmul(np.matmul(v1t, np.diag(L1[i])), p1), np.matmul(np.matmul(v2t, np.diag(L2[i])), p2)) for i in range(len(L1))], axis=0)

        return np.inner(q1,p1)*np.inner(q2,p2) + self.c*np.dot(L, Lamda).dot(R)

    def _row_normalized_adj(self, G,  edge_attr = None):
        """
        Get row normalized adjacency matrix

        Parameters
        --------------------------
        G - networkx graph

        Returns
        ---------------
        sparse csr matrix

        """

        # A = nx.linalg.adjacency_matrix(G, dtype = float)
        A = nx.adjacency_matrix(G ,weight=edge_attr)# scipy.sparse.csr_matrix(nx.adjacency_matrix(G ,weight=edge_attr), dtype=np.float64)
        if type(self.X[0]) == nx.classes.digraph.DiGraph:
            D_inv = scipy.sparse.dia_matrix(([1/float(d[1]) for d in G.out_degree()], 0), shape = (A.shape[0], A.shape[0]))
        else:
            D_inv = scipy.sparse.dia_matrix(([1/float(d[1]) for d in G.degree()], 0), shape = (A.shape[0], A.shape[0]))

        return A.dot(D_inv)


    def _edge_weights(self, G, edge_labels):
        """
        Create a weight matrix for labeled edges

        Parameters
        --------------------------
        G - networkx graph
        """
        pass





    def _normalized_adj(self, G, edge_attr = None):
        """
        Get normalized adjacency matrix

        Parameters
        --------------------------
        G - networkx graph

        """

        A = scipy.sparse.csr_matrix(nx.adjacency_matrix(G ,weight=edge_attr), dtype=np.float64)
        D_sq_inv = scipy.sparse.dia_matrix(([1/ np.sqrt(float(d[1])) for d in G.degree()], 0), shape = (A.shape[0], A.shape[0]))

        return D_sq_inv.dot(A).dot(D_sq_inv)

    def _get_adj_matrix(self, G, edge_attr = None):
        """
        Get adjacency matrix

        Parameters
        --------------------------
        G - networkx graph

        """
        return scipy.sparse.csr_matrix(nx.adjacency_matrix(G ,weight=edge_attr), dtype=np.float64)

    def _get_node_label_vectors(self, G, label_list, label_name = 'label'):
        """
        Get node label vectors

        1-D arrays are returned as we only need the diagonal information

        Parameters
        ---------------
        G - networkx graph
        label_list - list with all potential labels
        label_name - str, label name


        Returns
        -------------------------
        L - list of 1-d arrays

        
        """

        L = [] 

        for idx, label in enumerate(label_list):

            get_nodes_with_label = np.array(list(nx.get_node_attributes(G, label_name).values())) == label

            L.append(np.array(get_nodes_with_label, dtype = np.float64))

        return L


    def _get_label_adj(self, G, edge_labels, edge_labels_tag = 'sign', edge_attr = None):
        """

        Filter the adjacency matrix according to each label in edge_labels, w if edges have same label, 0 otherwise

        Parameter
        -------------------------
        G - networkx graph
        edge_labels - array with the label vocabulary
        edge_labels_tag - name of the edge label


        Returns
        --------------
        A - list of sparse matrices representing filtered adjacency matrices according to the labels in edge_labels
        
        """

        edge_attrs = nx.get_edge_attributes(G, edge_labels_tag )

        A = [None] * len(edge_labels)  # Store filtered adjacency matrices 

        for idx, label in enumerate(edge_labels):
            G_tmp = G.copy()
            for k, v in edge_attrs.items():
                if v != label:
                    G_tmp.remove_edge(k[0], k[1])

            A[idx] = scipy.sparse.csr_matrix(np.abs(nx.linalg.adjacency_matrix(G_tmp, weight = edge_attr)), dtype=np.float64)
        
        return A

    def _eigen_decomp(self, A, r):
        """
        Perform r eigenvalue decomposition on each matrix in A
        
        """

        nr_A = len(A)

        U = [None] * nr_A
        w = [None] * nr_A

        for i in range(nr_A):
            w[i], U[i] = eigsh(A[i].T, k = r)

        return w, U


    @staticmethod
    def normalize_gram_matrix(x):
        k = np.reciprocal(np.sqrt(np.diag(x)))
        k = np.resize(k, (len(k), 1))
        return np.multiply(x, np.outer(k,k))









