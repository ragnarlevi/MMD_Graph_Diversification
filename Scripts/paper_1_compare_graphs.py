import numpy as np
import pandas as pd

import datetime
import importlib
import os, sys
import tqdm
# from pandas_datareader import data
import networkx as nx
sys.path.insert(0, 'C:/Users/User/Code/MMDGraph')
sys.path.insert(0, 'C:/Users/ragna/Documents/Code/MMDGraph')

import pickle

from sklearn.covariance import graphical_lasso, GraphicalLasso, GraphicalLassoCV


from sklearn.covariance import GraphicalLassoCV
from sklearn.preprocessing import StandardScaler

import warnings


from multiprocessing import Pool, freeze_support



sys.path.insert(0, 'C:/Users/User/Code/MMDGraph/myKernels')
from myKernels import RandomWalk as rw
import MMDforGraphs as mg
import WL
import GNTK
import GraphStatKernel
import WWL



with open(f'data/paper/graph_construct_case_study_1_d_2_rw_weight_days_150.pkl', 'rb') as f:
    data_dict = pickle.load(f)
# Now perform test


def graph_test(n, graph_dict, k, dates ):
    print(k)
    nr_splits = 3
    m = n

    esg_return_df = pd.DataFrame()
    for group_1 in [0]:#range(nr_splits):
        for group_2 in [2]:#range(group_1+1, nr_splits):
            pbar = tqdm.tqdm( total=len(list(range(n, len(graph_dict[group_1]), 20))))
            for cnt, i in enumerate(range(n, len(graph_dict[group_1]), 20)):
                # print(i-cnt)

                G_data_sector = [ graph_dict[group_1][s] for s in range(cnt*20, i )] + [ graph_dict[group_2][s] for s in range(cnt*20, i )]
                calc_ok = True
                for c in [0.001, 0.0001, 0.00001, 0.000001]:

                    r = np.min((6, G_data_sector[0].number_of_nodes()-1))
                    rw_kernel = rw.RandomWalk(G_data_sector, c = c, normalize=0)
                    K = rw_kernel.fit_ARKU_plus(r = r, normalize_adj=False, verbose=False, edge_attr = 'weight')

                    v,_ = np.linalg.eigh(K)
                    v[np.abs(v) < 10e-5] = 0
                    if np.all(v) >= 0:
                        calc_ok = True
                        break
                    calc_ok = False

                if calc_ok != True:
                    print(f"{k} Kernel not psd")

                MMD_functions = [mg.MMD_b, mg.MMD_u, mg.MMD_l, mg.MONK_EST]
                kernel_hypothesis = mg.BoostrapMethods(MMD_functions)
                function_arguments = [dict(n = n, m = m ), 
                                    dict(n = n, m = m ),
                                    dict(n = n, m = m ), 
                                    dict(Q = 5, y1 = list(range(n)), y2 = list(range(n,n+n)) )]
                kernel_hypothesis.Bootstrap(K, function_arguments, B = 3000)

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
                info_dict['c'] = c
                info_dict['dates'] = dates[i]


                esg_return_df = pd.concat((esg_return_df, pd.DataFrame(info_dict, index = [0])), ignore_index=True)

                pbar.update()
    pbar.close()

    dates_out = dates[n:]
    return {'info_dict':esg_return_df, 'sector':k, 'n':n}



if __name__ == '__main__':


    with Pool(1) as pool:
        L = pool.starmap(graph_test, [(50, data_dict[i]['graph_dict'], 
                                        data_dict[i]['sector'], 
                                        data_dict[i]['dates']) for i in [6] ])#range(len(data_dict))])

    
    with open(f'data/paper/graph_test_case_study_1_d_2_rw_weight_days_150_window_50_util_robust.pkl', 'wb') as f:
            pickle.dump(L, f)
