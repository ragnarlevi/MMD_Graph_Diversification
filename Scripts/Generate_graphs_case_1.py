import numpy as np
import pandas as pd
import datetime
import os, sys
import tqdm
# from pandas_datareader import data
import networkx as nx
sys.path.insert(0, 'C:/Users/User/Code/MMD_Graph_Diversification')
from util import fetch_raw_data

import pickle

from sklearn.covariance import graphical_lasso, GraphicalLasso, GraphicalLassoCV


from sklearn.covariance import GraphicalLassoCV
from sklearn.preprocessing import StandardScaler

import warnings


from multiprocessing import Pool, freeze_support

# Load R packages
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
spectralGraphTopology = importr('spectralGraphTopology')
igraph = importr('igraph')
fingraph = importr('fingraph')
fitHeavyTail = importr('fitHeavyTail')
huge = importr('huge')


# Define function that ignores degree constraints for the LGMRF. Note this is a R function
robjects.r('''

library(spectralGraphTopology)

#' @export
#' @import spectralGraphTopology
learn_regular_heavytail_graph <- function(X,
                                          heavy_type = "gaussian", nu = NULL,
                                          w0 = "naive", d = 1,
                                          rho = 1, update_rho = TRUE, maxiter = 10000, reltol = 1e-5,
                                          verbose = TRUE) {
  X <- as.matrix(X)
  # number of nodes
  p <- ncol(X)
  # number of observations
  n <- nrow(X)
  LstarSq <- vector(mode = "list", length = n)
  for (i in 1:n)
    LstarSq[[i]] <- Lstar(X[i, ] %*% t(X[i, ])) / (n-1)
  # w-initialization
  w <- spectralGraphTopology:::w_init(w0, MASS::ginv(cor(X)))
  A0 <- A(w)
  A0 <- A0 / rowSums(A0)
  w <- spectralGraphTopology:::Ainv(A0)
  J <- matrix(1, p, p) / p
  # Theta-initilization
  Lw <- L(w)
  Theta <- Lw
  Y <- matrix(0, p, p)
  y <- rep(0, p)
  # ADMM constants
  mu <- 2
  tau <- 2
  # residual vectors
  primal_lap_residual <- c()
  primal_deg_residual <- c()
  dual_residual <- c()
  # augmented lagrangian vector
  lagrangian <- c()
  if (verbose)
    pb <- progress::progress_bar$new(format = "<:bar> :current/:total  eta: :eta",
                                     total = maxiter, clear = FALSE, width = 80)
  elapsed_time <- c()
  start_time <- proc.time()[3]
  for (i in 1:maxiter) {
    # update w
    LstarLw <- Lstar(Lw)
    DstarDw <- Dstar(diag(Lw))
    LstarSweighted <- rep(0, .5*p*(p-1))
    if (heavy_type == "student") {
      for (q in 1:n)
        LstarSweighted <- LstarSweighted + LstarSq[[q]] * compute_student_weights(w, LstarSq[[q]], p, nu)
    } else if(heavy_type == "gaussian") {
      for (q in 1:n)
        LstarSweighted <- LstarSweighted + LstarSq[[q]]
    }
    grad <- LstarSweighted - Lstar(rho * Theta + Y) + Dstar(y - rho * d) + rho * (LstarLw + DstarDw)
    eta <- 1 / (2*rho * (2*p - 1))
    wi <- w - eta * grad
    wi[wi < 0] <- 0
    Lwi <- L(wi)
    # update Theta
    eig <- eigen(rho * (Lwi + J) - Y, symmetric = TRUE)
    V <- eig$vectors
    gamma <- eig$values
    Thetai <- V %*% diag((gamma + sqrt(gamma^2 + 4 * rho)) / (2 * rho)) %*% t(V) - J
    # update Y
    R1 <- Thetai - Lwi
    Y <- Y + rho * R1
    # update y
    R2 <- diag(Lwi) - d
    #y <- y + rho * R2
    # compute primal, dual residuals, & lagrangian
    primal_lap_residual <- c(primal_lap_residual, norm(R1, "F"))
    primal_deg_residual <- c(primal_deg_residual, norm(R2, "2"))
    dual_residual <- c(dual_residual, rho*norm(Lstar(Theta - Thetai), "2"))
    lagrangian <- c(lagrangian, compute_augmented_lagrangian_ht(wi, LstarSq, Thetai, J, Y, y, d, heavy_type, n, p, rho, nu))
    # update rho
    if (update_rho) {
      s <- rho * norm(Lstar(Theta - Thetai), "2")
      r <- norm(R1, "F")
      if (r > mu * s)
        rho <- rho * tau
      else if (s > mu * r)
        rho <- rho / tau
    }
    if (verbose)
      pb$tick()
    has_converged <- (norm(Lw - Lwi, 'F') / norm(Lw, 'F') < reltol) && (i > 1)
    elapsed_time <- c(elapsed_time, proc.time()[3] - start_time)
    if (has_converged)
      break
    w <- wi
    Lw <- Lwi
    Theta <- Thetai
  }
  results <- list(laplacian = L(wi),
                  adjacency = A(wi),
                  theta = Thetai,
                  maxiter = i,
                  convergence = has_converged,
                  primal_lap_residual = primal_lap_residual,
                  primal_deg_residual = primal_deg_residual,
                  dual_residual = dual_residual,
                  lagrangian = lagrangian,
                  elapsed_time = elapsed_time)
  return(results)
}

compute_student_weights <- function(w, LstarSq, p, nu) {
  return((p + nu) / (sum(w * LstarSq) + nu))
}

compute_augmented_lagrangian_ht <- function(w, LstarSq, Theta, J, Y, y, d, heavy_type, n, p, rho, nu) {
  eig <- eigen(Theta + J, symmetric = TRUE, only.values = TRUE)$values
  Lw <- L(w)
  Dw <- diag(Lw)
  u_func <- 0
  if (heavy_type == "student") {
    for (q in 1:n)
      u_func <- u_func + (p + nu) * log(1 + n * sum(w * LstarSq[[q]]) / nu)
  } else if (heavy_type == "gaussian"){
    for (q in 1:n)
      u_func <- u_func + sum(n * w * LstarSq[[q]])
  }
  u_func <- u_func / n
  return(u_func - sum(log(eig)) + sum(y * (Dw - d)) + sum(diag(Y %*% (Theta - Lw)))
         + .5 * rho * (norm(Dw - d, "2")^2 + norm(Lw - Theta, "F")^2))
}

''')


learn_heavy_connected_graph = robjects.globalenv["learn_regular_heavytail_graph"]

print(os.getcwd())
# Load data
price_pivot, esg_pivot, sector_classification = fetch_raw_data("C:/Users/User/Code/MMD_Graph_Diversification/")

# esg smoothed stock
gp_esg_stock = pd.read_pickle('data/tidy/gp_esg_stock_data_frame.pkl')



def create_G(A:np.array):
  """
  Label graph
  A adjacency matrix as nx
  """
  G = nx.from_numpy_array(A)
  # set sign labels
  nx.set_edge_attributes(G, {(n1, n2): np.sign(val) for n1, n2, val in G.edges.data('weight')}, "sign")
  nodes_degree = dict(G.degree)
  # set degree as label
  nx.set_node_attributes(G, {key: str(value) for key, value in nodes_degree.items()}, "label")
  return G


def div_ratio(w, cov):
  # numerator is perfect correlation
  # denom is portfolio risk
  return np.inner(w, np.sqrt(np.diag(cov)))/np.sqrt(np.dot(w, cov).dot(w))

def var_div_ratio(w,data, q = 0.95):
  # w weights
  # d = data

  ind_var = []
  for col in range(data.shape[1]):
      ind_var.append(np.quantile(-data[:,col], q))

  port_var = np.quantile(np.dot(-data, w), q)

  return port_var/np.inner(ind_var,w)


# function to estimate covariance, and graphs

def graph_est(price_df, esg_df, all_stocks_in_sector, k,d = 1, window_size = 150, nr_splits = 3, graph_estimation = "lgmrf_heavy", scale = 'normal'):
  """
  Parameters
  ------------------------------
  price_df: pandas dataframe price data frame
  esg_df: pandas dataframe smoothed esg scores
  all_stocks_in_sector: array with names of stocks in the sector being analyzed
  k: str name of sector to analyze
  d: degree of LGMR graph generationh
  window_size: rolliwng window size. 
  nr_splits: How many splits to consdier the esg data. If 3 then the stocks will be splits into 3 categories, good, medium and bad. 
  Group number 0 containes the lowest esg scores
  graph_estimation:
  scale: if None no scaling

  """
  # information stored in dictionaries
  stocks_considered= dict()
  graph_dict = {i: [] for i in range(nr_splits)}
  max_sharpe_portfolio_dict = {i: [] for i in range(nr_splits)}
  GMV_portfolio_dict = {i: [] for i in range(nr_splits)}
  return_dict  = {i: [] for i in range(nr_splits)}
  cov_dict = {i: [] for i in range(nr_splits)}
  gmv_div_dict = {i: [] for i in range(nr_splits)}
  gmv_var_div_dict = {i: [] for i in range(nr_splits)}
  sharpe_div_dict = {i: [] for i in range(nr_splits)}
  sharpe_var_div_dict = {i: [] for i in range(nr_splits)}
  uni_div_dict = {i: [] for i in range(nr_splits)}
  uni_var_div_dict = {i: [] for i in range(nr_splits)}

  max_sharpe_portfolio_dict2 = {i: [] for i in range(nr_splits)}
  GMV_portfolio_dict2 = {i: [] for i in range(nr_splits)}
  return_dict2  = {i: [] for i in range(nr_splits)}
  cov_dict2 = {i: [] for i in range(nr_splits)}
  gmv_div_dict2 = {i: [] for i in range(nr_splits)}
  gmv_var_div_dict2 = {i: [] for i in range(nr_splits)}
  sharpe_div_dict2 = {i: [] for i in range(nr_splits)}
  sharpe_var_div_dict2 = {i: [] for i in range(nr_splits)}
  uni_div_dict2 = {i: [] for i in range(nr_splits)}
  uni_var_div_dict2 = {i: [] for i in range(nr_splits)}

  stock_partition = {i: [] for i in range(nr_splits)}
  
  dates4 = []

  for i in tqdm.tqdm(range(window_size, price_df.shape[0], 2)):
      

    # At first iteration determine which stocks in the sector will be considered.
    # If the total number of stocks in the sector are not divisible by nr_splits
    # then the first stocks which correspond to the reminder of the division will be omitted
    # in future iterations only the stocks in stocks_considered will be used
    if stocks_considered.get(k,None) is None:
      stocks_in_sector = price_df.columns[np.isin(price_df.columns,all_stocks_in_sector)]
      esg_stocks_in_sector = esg_df.columns[np.isin(esg_df.columns,all_stocks_in_sector)]

      stocks_considered[k] = stocks_in_sector.intersection(esg_stocks_in_sector)
      print(f'{k} has {len(stocks_considered[k])} stocks')
      

      if len(stocks_considered[k]) % nr_splits != 0:
        res = len(stocks_considered[k]) % nr_splits
        print(f'{k} dropped {stocks_considered[k][:res]}')
        stocks_considered[k] = stocks_considered[k][res:]


    # get esg scores of the stocks for the current iteration
    esg_i = np.array(esg_df[stocks_considered[k]].iloc[i])
    # order stocks
    stocks_ordered_i = np.array(stocks_considered[k][np.argsort(esg_i)])
    # get date of the iteration i
    date_i = esg_df.index[i]
    #NOT USED ------
    #  Find which dates are  in the current rolling window 
    #day_range_from_i = np.array([date_i - datetime.timedelta(days=i) for i in range(days_from_i)])
    # NOT USED END

    # Store date
    dates4.append(date_i)
    # create a dummy index so we can access the correct stocke later on
    stocks_indexes = np.array(range(len(stocks_considered[k])))
    # this for loop consideres each split and fetches the correct stocks for the split
    for i_split, stocks_index in enumerate(np.array_split(stocks_indexes, nr_splits)):

      # If nr_splits is 3 and number of stocks is 30, 
      # the at first iteration stocks_index = [0,2,...,9]
      # second iteration stocks_index = [10,2,...,19]
      # third iteration stocks_index = [20,2,...,29]

      # select the stocks in this split and only select the time for this rolling window
      stock_split_i = price_df[stocks_ordered_i[stocks_index]].iloc[(i-window_size):i]

      stock_partition[i_split].append(stocks_ordered_i[stocks_index])

      if i == window_size and i_split == 0:
        print(f"First iteration of {k}. Shape is {stock_split_i.shape}")

      X = np.array(stock_split_i)
      var = np.diag(np.cov(X.T))
      if scale == 'normal':
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
      

      try:
        if graph_estimation == 'lgmrf_heavy':
          # Find nu for multivariate t-student
          out_fit_mvt = fitHeavyTail.fit_mvt(X,nu="MLE-diag-resample")
          out_fit_mvt = dict(zip(out_fit_mvt.names, list(out_fit_mvt)))
          if d == 0:
            out = learn_heavy_connected_graph(X, heavy_type = "student", nu = out_fit_mvt['nu'][0], verbose = False)
            out = dict(zip(out.names, list(out)))
            precision_matrix = out['laplacian']
            G = create_G(out['adjacency'])
            graph_dict[i_split].append(G)
          else:
            out = fingraph.learn_regular_heavytail_graph(X, heavy_type = "student", nu = out_fit_mvt['nu'][0], d=d, verbose = False)
            out = dict(zip(out.names, list(out)))
            precision_matrix = out['laplacian']
            G = create_G(out['adjacency'])
            graph_dict[i_split].append(G)

        elif graph_estimation == 'lgmrf_normal':
          out = fingraph.learn_regular_heavytail_graph(X, heavy_type = "gaussian", d=d, verbose = False)
          out = dict(zip(out.names, list(out)))
          precision_matrix = out['laplacian']
          G = create_G(out['adjacency'])
          graph_dict[i_split].append(G)

        elif graph_estimation == 'sklearn_glasso':
          glasso = GraphicalLassoCV(cv=3).fit(X)
          precision_matrix = glasso.precision_
          precision_matrix_no_diag = precision_matrix.copy()
          np.fill_diagonal(precision_matrix_no_diag,0)
          G = create_G(-precision_matrix_no_diag)
          graph_dict[i_split].append(G)
        elif graph_estimation == 'huge_glasso_ebic':
          out = huge.huge(X, method = 'glasso', nlambda = 30,verbose = False)
          out_select = huge.huge_select(out, criterion = "ebic" )
          out_select = dict(zip(out_select.names, list(out_select)))
          precision_matrix = out_select['opt.icov'].copy()
          precision_matrix_no_diag = precision_matrix.copy()
          np.fill_diagonal(precision_matrix_no_diag,0)
          G = create_G(-precision_matrix_no_diag)
          graph_dict[i_split].append(G)

      except:
        # remove all graphs from this time point if there is a graph estimation failure for one of the splits.
        # To make sure the graphs are pairwise ordered correctly
        for remove_i in range(i_split):
          graph_dict[remove_i].pop(-1)

          # max_sharpe_portfolio_dict[remove_i].pop(-1)
          # GMV_portfolio_dict[remove_i].pop(-1)
          # return_dict[remove_i].pop(-1)
          # cov_dict[remove_i].pop(-1)
          
          # gmv_div_dict[remove_i].pop(-1)
          # gmv_var_div_dict[remove_i].pop(-1)
          # sharpe_div_dict[remove_i].pop(-1)
          # sharpe_var_div_dict[remove_i].pop(-1)
          # uni_div_dict[remove_i].pop(-1)
          # uni_var_div_dict[remove_i].pop(-1)

          # stock_partition[remove_i].pop(-1)

        dates4.pop(-1)
        break

      #scaler = StandardScaler()
      if graph_estimation == 'lgmrf_heavy' or graph_estimation == 'lgmrf_normal':
        precision_matrix = precision_matrix + 0.001*np.identity(precision_matrix.shape[0])
      
      # Calculate diversification using graph estimate
      S = np.linalg.inv(precision_matrix)
      cov_dict[i_split].append(S)
      mu = stock_split_i.mean()
      return_dict[i_split].append(mu)

      # GMV diversification
      w_gmv = np.dot(precision_matrix, np.ones(S.shape[0]))/np.dot(np.ones(S.shape[0]), precision_matrix).dot(np.ones(S.shape[0])) 
      GMV_portfolio_dict[i_split].append(w_gmv)
      gmv_div_dict[i_split].append(div_ratio(w_gmv,S))
      gmv_var_div_dict[i_split].append(var_div_ratio(w_gmv,X))
      # SHARPE
      w_sharpe = np.dot(precision_matrix, mu)/np.dot(np.ones(S.shape[0]), precision_matrix).dot(mu)
      max_sharpe_portfolio_dict[i_split].append(w_sharpe)
      sharpe_div_dict[i_split].append(div_ratio(w_sharpe,S))
      sharpe_var_div_dict[i_split].append(var_div_ratio(w_sharpe,X))
      # Uniform
      w_uni = np.ones(S.shape[1])/S.shape[1]
      uni_div_dict[i_split].append(div_ratio(w_uni,S))
      uni_var_div_dict[i_split].append(var_div_ratio(w_uni,X))

      # Calculate diversification by turning correlation back to covariance
      S = np.dot(np.diag(var), np.linalg.inv(precision_matrix)).dot(np.diag(var))
      precision_matrix = np.linalg.inv(precision_matrix)
      cov_dict2[i_split].append(S)
      mu = stock_split_i.mean()
      return_dict2[i_split].append(mu)

      # GMV diversification
      w_gmv = np.dot(precision_matrix, np.ones(S.shape[0]))/np.dot(np.ones(S.shape[0]), precision_matrix).dot(np.ones(S.shape[0])) 
      GMV_portfolio_dict2[i_split].append(w_gmv)
      gmv_div_dict2[i_split].append(div_ratio(w_gmv,S))
      gmv_var_div_dict2[i_split].append(var_div_ratio(w_gmv,X))
      # SHARPE
      w_sharpe = np.dot(precision_matrix, mu)/np.dot(np.ones(S.shape[0]), precision_matrix).dot(mu)
      max_sharpe_portfolio_dict2[i_split].append(w_sharpe)
      sharpe_div_dict2[i_split].append(div_ratio(w_sharpe,S))
      sharpe_var_div_dict2[i_split].append(var_div_ratio(w_sharpe,X))
      # Uniform
      w_uni = np.ones(S.shape[1])/S.shape[1]
      uni_div_dict2[i_split].append(div_ratio(w_uni,S))
      uni_var_div_dict2[i_split].append(var_div_ratio(w_uni,X))
          
  dates = np.array(dates4)

  assert len(graph_dict[0]) == len(graph_dict[1]), f"{k} 0 and 1 not same"
  assert len(graph_dict[0]) == len(graph_dict[2]), f"{k} 0 and 2 not same"
  assert len(graph_dict[1]) == len(graph_dict[2]), f"{k} 1 and 2 not same"


  return {'dates':dates, 'graph_dict':graph_dict, 'sector':k, 'cov_dict':cov_dict, 
  'GMV_portfolio_dict':GMV_portfolio_dict, 'max_sharpe_portfolio_dict':max_sharpe_portfolio_dict,
  'return_dict':return_dict, 'window_size':window_size, 'stock_partition':stock_partition, 'gmv_div_dict':gmv_div_dict,
  'gmv_var_div_dict':gmv_var_div_dict, 'sharpe_div_dict':sharpe_div_dict, 'sharpe_var_div_dict':sharpe_var_div_dict,
  'uni_div_dict':uni_div_dict, 'uni_var_div_dict':uni_var_div_dict,'cov_dict2':cov_dict2, 
  'GMV_portfolio_dict2':GMV_portfolio_dict2, 'max_sharpe_portfolio_dict2':max_sharpe_portfolio_dict2,
  'return_dict2':return_dict2,  'gmv_div_dict2':gmv_div_dict2,
  'gmv_var_div_dict2':gmv_var_div_dict2, 'sharpe_div_dict2':sharpe_div_dict2, 'sharpe_var_div_dict2':sharpe_var_div_dict2,
  'uni_div_dict2':uni_div_dict2, 'uni_var_div_dict2':uni_var_div_dict2}


if __name__ == '__main__':

  d = 1
  winow_len = 300
  graph_estimation = 'huge_glasso_ebic'
  scale = 'normal'

  print(graph_estimation)
  print(winow_len)
  print(scale)
  
  with Pool(1) as pool:
    L = pool.starmap(graph_est, [(price_pivot.loc[:, np.isin(price_pivot.columns,sector_classification[k])], 
                                  gp_esg_stock.loc[:, np.isin(gp_esg_stock.columns,sector_classification[k])], 
                                  sector_classification[k], 
                                    k, 
                                    d,
                                    winow_len, 
                                    3,
                                    graph_estimation) for k in ['Industrials']])#sector_classification.keys()

  with open(f'data/Graphs/case_study_1_d_{d}_winlen_{winow_len}_gest_{graph_estimation}_scale_{scale}.pkl', 'wb') as f:
    pickle.dump(L, f)


