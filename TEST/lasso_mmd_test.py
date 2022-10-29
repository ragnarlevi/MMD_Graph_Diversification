import networkx as nx
import numpy as np

import pickle # save data frame (results) in a .pkl file
import pandas as pd
import os, sys


import grakel as gk
import networkx as nx


sys.path.insert(0, 'C:/Users/User/Code/MMD_Graph_Diversification')
sys.path.insert(0, 'C:/Users/User/Code/MMD_Graph_Diversification/myKernels')
from myKernels import RandomWalk  as rw
import WL
from multiprocessing import Pool


import MMDforGraphs as mg
from sklearn.covariance import graphical_lasso
import argparse
import tqdm

import warnings
warnings.filterwarnings("ignore")










def run_samples_lasso(N, B, alpha, theta1, theta2, s1, s2, prec = False):
    import myKernels.RandomWalk as rw
    test_info = pd.DataFrame()
    k = theta1.shape[0]
    for sample in tqdm.tqdm(range(N)):

        Gs1 = []
        Gs2 = []
        error_1 = []
        error_2 = []
        error_1_mse = []
        error_2_mse = []
        n = 50

        for i in range(50):
            if prec:
                x1 = np.random.multivariate_normal(mean = np.zeros(k), cov = np.linalg.inv(theta1), size = 100)
            else:
                x1 = np.random.multivariate_normal(mean = np.zeros(k), cov = theta1, size = 100)
            A1 = np.corrcoef(x1.T)
            if alpha == 0:
                if prec:
                    A1 = np.linalg.inv(A1)
                    np.fill_diagonal(A1, 0) 
                    A1[np.abs(A1) < 1e-5] = 0
                else:
                    np.fill_diagonal(A1, 0) 
                    A1[np.abs(A1) < 1e-5] = 0
            else:
                gl = graphical_lasso(A1, alpha = alpha, max_iter = 1000)
                if prec:
                    A1 = gl[1]
                    A1[np.abs(A1) < 1e-5] = 0
                else:
                    A1 = gl[0]
                    A1[np.abs(A1) < 1e-5] = 0
                np.fill_diagonal(A1, 0)

            Gs1.append(nx.from_numpy_matrix(A1))
            error_1.append(error_func(theta1, A1))#.append(np.sum(np.logical_xor(np.abs(np.triu(A1,1)) > 0,np.abs(np.triu(theta1,1)) > 0)))
            error_1_mse.append(np.sum((A1 - theta1) ** 2))

            if prec:
                x2 = np.random.multivariate_normal(mean = np.zeros(k), cov = np.linalg.inv(theta2), size = 100)
            else:
                x2 = np.random.multivariate_normal(mean = np.zeros(k), cov =theta2, size = 100)
            
            A2 = np.corrcoef(x2.T)

            if alpha != 0:
                gl = graphical_lasso(A2, alpha = alpha, max_iter = 1000)
                np.fill_diagonal(A2, 0)
            Gs2.append(nx.from_numpy_matrix(A2))
            error_2.append(error_func(theta2, A2))#append(np.sum(np.logical_xor(np.abs(np.triu(A2,1)) > 0,np.abs(np.triu(theta2,1)) > 0)))
            error_2_mse.append(np.sum((A2 - theta2) ** 2))

        Gs = Gs1 + Gs2


        try:
            rw_kernel = rw.RandomWalk(Gs, c = 1e-3, normalize=0)
            K = rw_kernel.fit_ARKU_plus(r=6, verbose = False, edge_attr='weight')
            v,_ = np.linalg.eigh(K)
            # v[np.abs(v) < 10e-5] = 0
            if np.any(v < -10e-8):
                raise ValueError("Not psd WEIGHT")
        except:
            continue


        MMD_functions = [mg.MMD_u]

        kernel_hypothesis = mg.BoostrapMethods(MMD_functions)
        function_arguments=[dict(n = n, m = n )]
        kernel_hypothesis.Bootstrap(K, function_arguments, B = B)


        test_info = pd.concat((test_info, pd.DataFrame({
            'p_val':kernel_hypothesis.p_values['MMD_u'],
            'sample':sample,
            'mean_error_1':np.mean(error_1),
            'mean_error_2':np.mean(error_2),
            'mean_mse_error_1':np.mean(error_1_mse),
            'mean_mse_error_2':np.mean(error_2_mse),
            'alpha':alpha,
            's1':s1,
            's2':s2,
            'kernel':'sp',
            'prec':prec


        }, index = [0])), ignore_index=True)



    return test_info


def error_func(theta, A):

    theta_vec = theta[np.triu_indices(theta.shape[0], k = 1)]
    A_vec = A[np.triu_indices(A.shape[0], k = 1)]

    error_count = 0.0
    for i in range(len(theta_vec)):

        if np.sign(theta_vec[i]) != np.sign(A_vec[i]):
            error_count += 1.0

        # if (theta_vec[i] > 0) & (A_vec[i] <= 0) :
        #     error_count += 1.0
        # elif (theta_vec[i] < 0) & (A_vec[i] >= 0) :
        #     error_count += 1.0
        # elif (theta_vec[i] == 0) & (np.abs(A_vec[i]) > 0):
        #     error_count += 1.0

    return error_count




def gen_theta(k, sparsity, seed):
    np.random.seed(seed=seed)
    # generate the symmetric sparsity mask
    mask = np.random.uniform(size = k)
    mask = mask * (mask < sparsity)
    mask = np.triu(mask)
    mask = mask + mask.T + np.identity(k)
    mask[mask > 0] = 1

    # generate the symmetric precision matrix
    theta = np.random.normal(size = (k,k))
    theta = np.random.normal(size = (k,k))
    theta = np.triu(theta)
    theta = theta + theta.T + np.identity(k)

    # apply the reqired sparsity
    theta = theta * mask

    l, _ = np.linalg.eigh(theta)
    # force it to be positive definite
    theta = theta - (np.min(l)-.1) * np.identity(k)

    return theta



def run_samples_lasso2(N, A, A_s, alpha,H0):
    from sklearn.covariance import graphical_lasso, GraphicalLasso

    test_info = pd.DataFrame()
    k = A.shape[0]
    for sample in tqdm.tqdm(range(N)):
        Gs1 = []
        Gs2 = []
        error_1_mse = []
        error_2_mse = []
        error_1 = []
        error_2 = []

        for i in range(50):
            x1 = np.random.multivariate_normal(mean = np.zeros(k), cov = np.linalg.inv(A), size = 100)
            gl = GraphicalLasso(alpha = alpha, max_iter = 1000).fit(x1)
            A1 = gl.precision_
            Gs1.append(nx.from_numpy_matrix(A1))
            error_1_mse.append(np.sum(np.abs(A1 - A)))
            error_1.append(error_func(A, A1))

            x2 = np.random.multivariate_normal(mean = np.zeros(k), cov = np.linalg.inv(A_s), size = 100)
            gl = GraphicalLasso(alpha = alpha, max_iter = 1000).fit(x2)
            A2 = gl.precision_
            Gs2.append(nx.from_numpy_matrix(A2))
            error_2_mse.append(np.sum(np.abs(A2 - A_s)))
            error_2.append(error_func(A_s, A2))

        Gs = Gs1 + Gs2


        rw_kernel = rw.RandomWalk(Gs, c = 1e-3, normalize=0)
        K = rw_kernel.fit_ARKU_plus(r=6, verbose = False, edge_attr='weight')
        v,_ = np.linalg.eigh(K)
        # v[np.abs(v) < 10e-5] = 0
        if np.any(v < -10e-8):
            raise ValueError("Not psd WEIGHT")


        MMD_functions = [mg.MMD_u]

        kernel_hypothesis = mg.BoostrapMethods(MMD_functions)
        function_arguments=[dict(n = 50, m = 50 )]
        kernel_hypothesis.Bootstrap(K, function_arguments, B = 5000)
        kernel_hypothesis.p_values


        test_info = pd.concat((test_info, pd.DataFrame({
                'p_val':kernel_hypothesis.p_values['MMD_u'],
                'sample':sample,
                'mean_error_1':np.mean(error_1),
                'mean_error_2':np.mean(error_2),
                'mean_mse_error_1':np.mean(error_1_mse),
                'mean_mse_error_2':np.mean(error_2_mse),
                'alpha':alpha,
                'kernel':'rw',
                'H0':H0


            }, index = [0])), ignore_index=True)



    return test_info



if __name__ == '__main__':






    # parser = argparse.ArgumentParser()
    # parser.add_argument('-prec', '--prec', type=int,metavar='', help='Precision?')



    # args = parser.parse_args()
    # prec = bool(args.prec)


    # s1 = 0.5
    # s2 = 0.6
    # theta1 = gen_theta(11, s1, 42)
    # theta2 = gen_theta(11, s2, 42)

    # print(theta1)
    # print(np.allclose(theta1, theta2))
    
    
    # for alpha in np.linspace(start = 0, stop = 1, num = 50):
    #     print(alpha)
    #     N = 150
    #     B = 10000



    #     with Pool(6) as pool:
    #         L = pool.starmap(run_samples_lasso, [(N, B, alpha, theta1, theta2, s1, s2, prec), 
    #                                                 (N, B, alpha, theta1, theta2, s1, s2, prec),
    #                                                 (N, B, alpha, theta1, theta2, s1, s2, prec),
    #                                                 (N, B, alpha, theta1, theta2, s1, s2, prec),
    #                                                 (N, B, alpha, theta1, theta2, s1, s2, prec),
    #                                                 (N, B, alpha, theta1, theta2, s1, s2, prec)])
            

    #         df = pd.concat(L)


    #     with open(f'data/GLasso/alpha_{alpha}_{s1}_{s2}_prec_{prec}_kernel_sp.pkl', 'wb') as f:
    #         pickle.dump(df, f)



    # s1 = 0.5
    # s2 = 0.5
    # theta1 = gen_theta(11, s1, 42)
    # theta2 = gen_theta(11, s2, 42)
    # print(theta1)
    # print(np.allclose(theta1, theta2))

    # for alpha in np.linspace(start = 0, stop = 1, num = 50):
    #     print(alpha)
    #     N = 150
    #     B = 10000



    #     with Pool(6) as pool:
    #         L = pool.starmap(run_samples_lasso, [(N, B, alpha, theta1, theta2, s1, s2, prec), 
    #                                                 (N, B, alpha, theta1, theta2, s1, s2, prec),
    #                                                 (N, B, alpha, theta1, theta2, s1, s2, prec),
    #                                                 (N, B, alpha, theta1, theta2, s1, s2, prec),
    #                                                 (N, B, alpha, theta1, theta2, s1, s2, prec),
    #                                                 (N, B, alpha, theta1, theta2, s1, s2, prec)
    #                                                 ])
            

    #         df = pd.concat(L)


    #     with open(f'data/GLasso/alpha_{alpha}_{s1}_{s2}_prec_{prec}_kernel_sp.pkl', 'wb') as f:
    #         pickle.dump(df, f)


        
    G = nx.fast_gnp_random_graph(11, 0.4, seed = 42)

    for e in G.edges():
        if np.random.uniform() <0.1:
            w = np.random.uniform(low = 0.1, high = 0.3)
            G.edges[e[0], e[1]]['weight'] = -w
        else:
            w = np.random.uniform(low = 0.1, high = 0.3)
            G.edges[e[0], e[1]]['weight'] = w

    A = np.array(nx.adjacency_matrix(G).todense())
    np.fill_diagonal(A, np.sum(np.abs(A), axis = 1))

    A_s = A.copy()
    A_s[6,4] = 0
    A_s[4,6] = 0
    A_s[5,0] = 0
    A_s[0,5] = 0



    for alpha in np.linspace(start = 0.0, stop = 0.3, num = 60):
        print(alpha)
        N = 150




        with Pool(6) as pool:
            L = pool.starmap(run_samples_lasso2, [(N, A, A_s, alpha, False),
                                                  (N, A, A_s, alpha, False),
                                                  (N, A, A_s, alpha, False),
                                                  (N, A, A_s, alpha, False),
                                                  (N, A, A_s, alpha, False),
                                                  (N, A, A_s, alpha, False)])
            

            df = pd.concat(L)


        with open(f'data/GLasso3/alpha_{alpha}_H0_false_kernel_RW.pkl', 'wb') as f:
            pickle.dump(df, f)



    A_s = A.copy()


    for alpha in np.linspace(start = 0, stop = 0.3, num = 60):
        print(alpha)
        N = 150




        with Pool(6) as pool:
            L = pool.starmap(run_samples_lasso2, [(N, A, A_s, alpha, True),
                                                  (N, A, A_s, alpha, True),
                                                  (N, A, A_s, alpha, True),
                                                  (N, A, A_s, alpha, True),
                                                  (N, A, A_s, alpha, True),
                                                  (N, A, A_s, alpha, True)])
            

            df = pd.concat(L)


        with open(f'data/GLasso3/alpha_{alpha}_H0_true_kernel_RW.pkl', 'wb') as f:
            pickle.dump(df, f)