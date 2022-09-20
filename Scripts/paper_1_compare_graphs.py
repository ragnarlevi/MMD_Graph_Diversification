from operator import truediv
import numpy as np
import pandas as pd

import datetime
import importlib
import os, sys
import tqdm
# from pandas_datareader import data
import networkx as nx
sys.path.insert(0, 'C:/Users/User/Code/MMD_Graph_Diversification')
import pickle

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


def graph_test(study, transform, scale, n,graph_name, ptype, B, edge_attr = 'weight',day_step = 1, graph_label = None, weight_fun = None, c = 0.001):
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

    file = f'data/Graphs/{study}_d_1_winlen_300_gest_huge_glasso_ebic_scale_{scale}_trans_{transform}.pkl'
    with open(file, 'rb') as f:
        data_dict = pickle.load(f)

    nr_splits = 3
    m = n

    graph_dict = data_dict[graph_name]
    weights = data_dict['portfolios_info'][ptype]['weights']
    k = data_dict['sector']
    dates = data_dict['dates']
    print(k)




    esg_return_df = pd.DataFrame()
    for group_1 in [0]:#range(nr_splits):
        for group_2 in [2]:#range(group_1+1, nr_splits):
            total_length = len(graph_dict[group_1])
            pbar = tqdm.tqdm( total=len(list(range(n, total_length, day_step))))
            for cnt, i in enumerate(range(n, total_length, day_step)):
                # print(i-cnt)n


                Gs = [ graph_dict[group_1][s] for s in range(cnt*day_step, i )] + [ graph_dict[group_2][s] for s in range(cnt*day_step, i )]
                
                # Gs = [ nx.from_numpy_array(nx.adjacency_matrix(graph_dict[group_1][s]).todense()/1000) for s in range(cnt*day_step, i )] + [ nx.from_numpy_array(nx.adjacency_matrix(graph_dict[group_2][s]).todense()/1000) for s in range(cnt*day_step, i )]
                if edge_attr == 'weight' and weight_fun == "abs":
                    Gs = [nx.from_numpy_array(np.abs(nx.adjacency_matrix(Gs[k]).todense())) for k in range(len(Gs))]

                # rw nr eigenvalues
                r = np.min((6, Gs[0].number_of_nodes()-1))

                if ptype is None:
                    p= None
                else:
                    p = np.vstack(([ weights[group_1][s] for s in range(cnt*day_step, i )],[ weights[group_2][s] for s in range(cnt*day_step, i )]))

                
                if graph_label is None:
                    rw_kernel = rw.RandomWalk(Gs, c = c, normalize=0, p=p)
                    K = rw_kernel.fit_ARKU_plus(r = r, normalize_adj=False, verbose=False, edge_attr = edge_attr)

                    v,_ = np.linalg.eigh(K)
                    # v[np.abs(v) < 10e-5] = 0
                    if np.any(v < -10e-8):
                        raise ValueError("Not psd")
                elif graph_label == 'signed':


                    rw_kernel = rw.RandomWalk(Gs, c = c, normalize=0, p =p)
                    K = rw_kernel.fit_ARKU_edge(r = r, edge_labels = [1,-1], verbose=False, edge_attr = edge_attr, edge_label_tag='sign')

                    v,_ = np.linalg.eigh(K)
                    #v[np.abs(v) < 10e-5] = 0
                    if np.any(v < -10e-8):
                        raise ValueError("Not psd")

                else:
                    ValueError(f"Check if graph_type is written correctly")



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
                info_dict['c'] = c
                info_dict['dates'] = dates[i]
                info_dict['dates_mid'] = dates[int((i+(i-n))/2)]


                esg_return_df = pd.concat((esg_return_df, pd.DataFrame(info_dict, index = [0])), ignore_index=True)

                pbar.update()
    pbar.close()

    L= {'info_dict':esg_return_df, 'sector':k, 'n':n, 'dates':dates[n:], 'graph_dict':graph_name, 'scale':data_dict['scale'],
    'transform':data_dict['transform'], 'transform':transform, 'scale':scale}

    with open(f'data/mmd_test/{study}_d_{1}_winlen_{300}_gest_{"huge_glasso_ebic"}_scale_{scale}_trans_{transform}_gname_{graph_name}_n_{n}_B_{B}_rw_{edge_attr}_dstep_{day_step}_glabel_{graph_label}_wfun_{weight_fun}_p_{ptype}.pkl', 'wb') as f:
        pickle.dump(L, f)

    return None




if __name__ == '__main__':

    d = 1
    winow_len = 300
    graph_estimation = 'huge_glasso_ebic'
    edge_attr = 'weight'

    n = 20
    day_step = 5
    graph_label = None#'signed'
    weight_fun = None
    B = 5000





    with Pool(3) as pool:
        L = pool.starmap(graph_test, [( study, transform, scale,
                                        n, 
                                        'graph_dict',
                                        ptype,
                                        B,
                                        edge_attr,
                                        day_step,
                                        graph_label,
                                        weight_fun,
                                        1e-10) for study, transform, scale, ptype in [
                                                                                      ('Communication Services', 'nonparanormal', False, 'uniform'),
                                                                                      ('Communication Services', 'nonparanormal', False, 'sharpe'),
                                                                                      ('Communication Services', 'nonparanormal', False, 'gmv')] ])

    
