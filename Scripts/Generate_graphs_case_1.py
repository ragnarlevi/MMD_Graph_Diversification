import numpy as np
import pandas as pd
import datetime
import os, sys
import warnings
import tqdm
# from pandas_datareader import data
import networkx as nx
sys.path.insert(0, 'C:/Users/User/Code/MMD_Graph_Diversification')
from util import fetch_raw_data

import pickle

from sklearn.covariance import graphical_lasso, GraphicalLasso, GraphicalLassoCV


from sklearn.covariance import GraphicalLassoCV
from sklearn.preprocessing import StandardScaler
from statsmodels.distributions.empirical_distribution import ECDF
import scipy.integrate as integrate

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



robjects.r('''

library(huge)

huge.glasso = function(x, scale = FALSE, lambda = NULL, lambda.min.ratio = NULL, nlambda = NULL, scr = NULL, cov.output = FALSE, verbose = TRUE){

  gcinfo(FALSE)
  n = nrow(x)
  d = ncol(x)
  cov.input = isSymmetric(x)
  if(cov.input)
  {
    if(verbose) cat("The input is identified as the covariance matrix.\n")
    S = x
  }
  else
  {
    if(scale){
      print("SCALE")
      x = scale(x)
    S = cor(x)
    }else{
    print("my_method")
    S = cov(x)
    }

  }
  rm(x)
  gc()
  if(is.null(scr)) scr = FALSE
  if(!is.null(lambda)) nlambda = length(lambda)
  if(is.null(lambda))
  {
    if(is.null(nlambda))
      nlambda = 10
    if(is.null(lambda.min.ratio))
      lambda.min.ratio = 0.1
    lambda.max = max(max(S-diag(d)),-min(S-diag(d)))
    lambda.min = lambda.min.ratio*lambda.max
    lambda = exp(seq(log(lambda.max), log(lambda.min), length = nlambda))
  }

  fit = .Call("_huge_hugeglasso",S,lambda,scr,verbose,cov.output,PACKAGE="huge")

  fit$scr = scr
  fit$lambda = lambda
  fit$cov.input = cov.input
  fit$cov.output = cov.output

  rm(S)
  gc()
  if(verbose){
       cat("\nConducting the graphical lasso (glasso)....done.                                          \r")
       cat("\n")
      flush.console()
  }
  return(fit)
}

huge = function(x, scale = scale, lambda = NULL, nlambda = NULL, lambda.min.ratio = NULL, method = "mb", scr = NULL, scr.num = NULL, cov.output = FALSE, sym = "or", verbose = TRUE)
{
	gcinfo(FALSE)
	est = list()
	est$method = method

	if(method == "ct")
	{
		fit = huge.ct(x, nlambda = nlambda, lambda.min.ratio = lambda.min.ratio, lambda = lambda, verbose = verbose)
		est$path = fit$path
		est$lambda = fit$lambda
		est$sparsity = fit$sparsity
		est$cov.input = fit$cov.input
		rm(fit)
		gc()
	}

	if(method == "mb")
	{
		fit = huge.mb(x, lambda = lambda, nlambda = nlambda, lambda.min.ratio = lambda.min.ratio, scr = scr, scr.num = scr.num, sym = sym, verbose = verbose)
		est$path = fit$path
		est$beta = fit$beta
		est$lambda = fit$lambda
		est$sparsity = fit$sparsity
		est$df = fit$df
		est$idx_mat = fit$idx_mat
		est$sym = sym
		est$scr = fit$scr
		est$cov.input = fit$cov.input
		rm(fit,sym)
		gc()
	}


	if(method == "glasso")
	{
		fit = huge.glasso(x, scale = scale, nlambda = nlambda, lambda.min.ratio = lambda.min.ratio, lambda = lambda, scr = scr, cov.output = cov.output, verbose = verbose)
		est$path = fit$path
		est$lambda = fit$lambda
		est$icov = fit$icov
		est$df = fit$df
		est$sparsity = fit$sparsity
		est$loglik = fit$loglik
		if(cov.output)
			est$cov = fit$cov
		est$cov.input = fit$cov.input
		est$cov.output = fit$cov.output
		est$scr = fit$scr
		rm(fit)
		gc()
	}

	if(method == "tiger")
	{
	  fit = huge.tiger(x, lambda = lambda, nlambda = nlambda, lambda.min.ratio = lambda.min.ratio, sym = sym, verbose = verbose)
	  est$path = fit$path
	  est$lambda = fit$lambda
	  est$sparsity = fit$sparsity
	  est$df = fit$df
	  est$idx_mat = fit$idx_mat
	  est$sym = sym
	  est$scr = fit$scr
	  est$cov.input = fit$cov.input
	  est$icov = fit$icov;
	  rm(fit,sym)
	  gc()
	}

	est$data = x

	rm(x,scr,lambda,lambda.min.ratio,nlambda,cov.output,verbose)
	gc()
	class(est) = "huge"
	return(est)
}

my_huge <- function(X, gamma = 0.1, lamda = exp(seq(log(1e-3), log(1e-7), length = 50)),scale = FALSE){

    out.glasso = huge(X, scale = scale, lambda = lamda, method = "glasso")
    return(huge.select(out.glasso, criterion = "ebic",ebic.gamma = 0.1 ))


}



''')

my_huge = robjects.globalenv["my_huge"]

def get_index(tick):
    """
    Function that takes the sp500 index from yahoo
    """
    
    import requests
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
    # ESG historical data (only changes yearly)
    url_esg = f"https://query1.finance.yahoo.com/v7/finance/spark?symbols={tick}&range=10y&interval=1d&indicators=close&includeTimestamps=false&includePrePost=false&corsDomain=finance.yahoo.com&.tsrc=finance"
    response = requests.get(url_esg, headers=headers)
    if response.ok:
        sp500 = pd.DataFrame({'date':pd.to_datetime(response.json()['spark']['result'][0]['response'][0]['timestamp'], unit= 's'),
                              'price':response.json()['spark']['result'][0]['response'][0]['indicators']['quote'][0]['close']})
    
    else:
        print("Empty data frame")
        sp500 = pd.DataFrame()



    return sp500

sp500 = get_index('^GSPC')
sp500['date'] = pd.to_datetime(sp500['date']).dt.date
sp500['return'] = 1 + sp500['price'].pct_change()
sp500['log_return'] = np.log(sp500['price']).diff()
sp500 = sp500.iloc[:,:].dropna(axis= 0)

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

def fix_weight(w:np.array):

    if np.sum(w[w <0]) <-0.3:

        # fix negative
        w[w <0] = 0.3*w[w <0]/np.abs(np.sum(w[w <0]))
        w[w >=0] = 1.3*w[w >=0]/np.abs(np.sum(w[w >=0]))

    return w

def omega(x, level = 0):
  ecdf = ECDF(x)  
  numerator = integrate.quad(lambda x: 1-ecdf(x), level, np.inf, limit = 10000)
  denominator = integrate.quad(ecdf, -np.inf, level, limit = 10000)
  if denominator[0] == 0.0:
    return 10
  else:
    return numerator[0]/denominator[0]

def sharpe(mu, sigma, r_f = 0):

  return (mu-r_f)/sigma

def sortino(mu,x, r_f = 0):

  x_above = x.copy()
  x_above[x_above > r_f] = r_f

  return (mu -r_f)/(np.sqrt(np.mean(x_above ** 2)))

def beta(X_port, X_index):
  return np.cov(X_port,X_index)[0,1]/np.var(X_index)

def treynor(mu, beta, r_f = 0):
  return (mu-r_f)/beta

def max_drawdown(X_port):
  x = np.cumprod(X_port)
  return np.min((x/np.array(pd.DataFrame(x).cummax().iloc[:,0])-1))


def portfolio(S,precision_matrix, mu, stock_split_i, type):


  if type == 'uniform':
      w = np.ones(S.shape[1])/S.shape[1]
      mu_p = np.mean(np.dot(stock_split_i, w))
      var_p = np.dot(w,S).dot(w)
  elif type == 'sharpe':
      w = np.dot(precision_matrix, mu)/np.dot(np.ones(S.shape[0]), precision_matrix).dot(mu) 
      w = fix_weight(w)
      mu_p = np.mean(np.dot(stock_split_i, w))
      var_p = np.dot(w,S).dot(w)
  elif type == 'gmv':
      w = np.dot(precision_matrix, np.ones(S.shape[0]))/np.dot(np.ones(S.shape[0]), precision_matrix).dot(np.ones(S.shape[0])) 
      w = fix_weight(w)
      mu_p = np.mean(np.dot(stock_split_i, w))
      var_p = np.dot(w,S).dot(w)


  return w, mu_p, var_p 


# function to estimate covariance, and graphs

def graph_est(price_df, esg_df, all_stocks_in_sector, k,d = 1, window_size = 150, nr_splits = 3, graph_estimation = "lgmrf_heavy", scale = False, transform = None, lamda = [1, 0.1]):
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
  return_dict  = {i: [] for i in range(nr_splits)}
  esg_mean = {i: [] for i in range(nr_splits)}
  esg_std = {i: [] for i in range(nr_splits)}
  esg_max = {i: [] for i in range(nr_splits)}
  esg_min = {i: [] for i in range(nr_splits)}
  cov_dict = {i: [] for i in range(nr_splits)}
  prec_dict = {i: [] for i in range(nr_splits)}
  prec_dict_plus_1 = {i: [] for i in range(nr_splits)}
  prec_dict_minus_1 = {i: [] for i in range(nr_splits)}
  opt_lambda = {i: [] for i in range(nr_splits)}
  reg_lambda = {i: [] for i in range(nr_splits)}

  portfolios_info = {}
  portfolios_reg_info = {}

  for port_type in ['uniform', 'sharpe', 'gmv']:
    portfolios_info[port_type] = {}
    portfolios_info[port_type]['weights'] = {i: [] for i in range(nr_splits)}
    portfolios_info[port_type]['cov_div'] = {i: [] for i in range(nr_splits)}
    portfolios_info[port_type]['var_div'] = {i: [] for i in range(nr_splits)}
    portfolios_info[port_type]['omega'] = {i: [] for i in range(nr_splits)}
    portfolios_info[port_type]['sharpe'] = {i: [] for i in range(nr_splits)}
    portfolios_info[port_type]['sortino'] = {i: [] for i in range(nr_splits)}
    portfolios_info[port_type]['beta'] = {i: [] for i in range(nr_splits)}
    portfolios_info[port_type]['treynor'] = {i: [] for i in range(nr_splits)}
    portfolios_info[port_type]['max_draw'] = {i: [] for i in range(nr_splits)}

  nlambda = len(lamda)
  reg_interval = np.concatenate((np.arange(0, nlambda, 10), [nlambda-1]))
  for port_type in ['uniform', 'sharpe', 'gmv']:
    portfolios_reg_info[port_type] = {}
    for j in reg_interval:
      j = int(j)
      portfolios_reg_info[port_type][j] = {}
      portfolios_reg_info[port_type][j]['weights'] = {i: [] for i in range(nr_splits)}
      portfolios_reg_info[port_type][j]['cov_div'] = {i: [] for i in range(nr_splits)}
      portfolios_reg_info[port_type][j]['var_div'] = {i: [] for i in range(nr_splits)}
      portfolios_reg_info[port_type][j]['omega'] = {i: [] for i in range(nr_splits)}
      portfolios_reg_info[port_type][j]['sharpe'] = {i: [] for i in range(nr_splits)}
      portfolios_reg_info[port_type][j]['sortino'] = {i: [] for i in range(nr_splits)}
      portfolios_reg_info[port_type][j]['beta'] = {i: [] for i in range(nr_splits)}
      portfolios_reg_info[port_type][j]['treynor'] = {i: [] for i in range(nr_splits)}
      portfolios_reg_info[port_type][j]['max_draw'] = {i: [] for i in range(nr_splits)}
      


  stock_partition = {i: [] for i in range(nr_splits)}
  
  dates4 = []
  min_max_date = np.min([np.max(price_df.index), np.max(esg_df.index)])
  price_df=price_df.loc[price_df.index <= min_max_date]
  for i in tqdm.tqdm(range(window_size, price_df.shape[0], 2)): #price_df.shape[0]
      

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
    esg_i = np.array(esg_df[stocks_considered[k]].loc[esg_df.index == price_df.index[i]].iloc[0,:])

    # get snp500 index for current iteration, used in  
    snp500_i = np.array(sp500.loc[np.isin(sp500.date,price_pivot.iloc[(i-window_size):i].index), 'log_return'])
    # order stocks
    stocks_ordered_i = np.array(stocks_considered[k][np.argsort(esg_i)])
    # get date of the iteration i
    date_i = price_df.index[i]


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
      
      if transform is None:
        stock_split_i = np.array(price_df[stocks_ordered_i[stocks_index]].iloc[(i-window_size):i])
      elif transform == 'nonparanormal':
        stock_split_i = price_df[stocks_ordered_i[stocks_index]].iloc[(i-window_size):i]
        mu_tmp = np.array(stock_split_i.mean())
        S_tmp = np.array(stock_split_i.cov())
        stock_split_i = huge.huge_npn(np.array(stock_split_i) , npn_func="truncation")
        # convert to correct scale again
        stock_split_i = (stock_split_i*np.sqrt(np.diag(S_tmp)) + mu_tmp)
      else:
        raise ValueError(f"No transform called {transform}")
      
      # get standar deviation of each asset
      var = np.diag(np.cov(stock_split_i.T))
      if i == window_size:
        print(var.shape)

      esg_split_i_mean = esg_df[stocks_ordered_i[stocks_index]].iloc[i].mean()
      esg_mean[i_split].append(esg_split_i_mean)
      esg_split_i_std = esg_df[stocks_ordered_i[stocks_index]].iloc[i].std() 
      esg_std[i_split].append(esg_split_i_std)
      esg_split_i_max = esg_df[stocks_ordered_i[stocks_index]].iloc[i].max() 
      esg_max[i_split].append(esg_split_i_max)
      esg_split_i_min = esg_df[stocks_ordered_i[stocks_index]].iloc[i].min() 
      esg_min[i_split].append(esg_split_i_min)

      stock_partition[i_split].append(stocks_ordered_i[stocks_index])

      if i == window_size and i_split == 0:
        print(f"First iteration of {k}. Shape is {stock_split_i.shape}")

      X = np.array(stock_split_i)
      
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
          # out = huge.huge(X, method = 'glasso', nlambda = nlambda,verbose = False, lambda_min_ratio = 0.001)
          # out_select = huge.huge_select(out, criterion = "ebic",ebic_gamma = 0.1 )
          # out_select = dict(zip(out_select.names, list(out_select)))
          out_select = my_huge(X, gamma = 0.01, lamda = lamda, scale = scale)
          out_select = dict(zip(out_select.names, list(out_select)))
          where_optimal = int(np.where(out_select['opt.lambda'][0] == out_select['lambda'])[0][0])
          print(where_optimal)
          if (where_optimal == nlambda-1) or (where_optimal == 0):
            warnings.warn("Warning optimal graph is last or first regularization parameter")
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
      

      opt_lambda[i_split].append(out_select['opt.lambda'][0])
      reg_lambda[i_split].append(out_select['lambda'][reg_interval])
      # Calculate diversification using graph estimate
      S = np.linalg.inv(precision_matrix)
      # If X was scaled during graph construction
      # we need to get the covariance back (we do not want to use correlation for portfolio optimization)
      if scale:
        S = np.dot(np.diag(np.sqrt(var)), S).dot(np.diag(np.sqrt(var)))
      cov_dict[i_split].append(S)
      mu = np.mean(stock_split_i, axis=0)
      return_dict[i_split].append(mu)
      prec_dict[i_split].append(precision_matrix)
      if (where_optimal+1)<=(nlambda-1):
        prec_dict_plus_1[i_split].append(out_select['icov'][where_optimal+1])
      else:
        prec_dict_plus_1[i_split].append(np.nan)
      if (where_optimal-1)>=0:
        prec_dict_minus_1[i_split].append(out_select['icov'][where_optimal-1])
      else:
         prec_dict_minus_1[i_split].append(np.nan)

      for port_type in ['uniform', 'sharpe', 'gmv']:
        w, mu_p, var_p = portfolio(S, np.linalg.inv(S), mu, np.array(stock_split_i), port_type)
        r_p = np.dot(np.array(stock_split_i),w)
        # print(np.sum(r_p <0))
        portfolios_info[port_type]['weights'][i_split].append(w)
        portfolios_info[port_type]['cov_div'][i_split].append(div_ratio(w,S))
        portfolios_info[port_type]['var_div'][i_split].append(var_div_ratio(w,np.array(stock_split_i)))
        portfolios_info[port_type]['sharpe'][i_split].append(sharpe(mu_p, np.sqrt(var_p)))
        portfolios_info[port_type]['omega'][i_split].append(omega(r_p))
        portfolios_info[port_type]['sortino'][i_split].append(sortino(mu_p, r_p))

        beta_p = beta(r_p, snp500_i)
        portfolios_info[port_type]['beta'][i_split].append(beta_p)
        portfolios_info[port_type]['treynor'][i_split].append(treynor(mu_p, beta_p))


      
      # for j in reg_interval:
      #   j = int(j)
      #   for port_type in ['uniform', 'sharpe', 'gmv']:
      #     S = np.linalg.inv(out_select['icov'][j])
      #     if scale:
      #       S = np.dot(np.diag(var), S).dot(np.diag(var))
      #     w, mu_p, var_p = portfolio(S, np.linalg.inv(S), mu, np.array(stock_split_i), port_type)
      #     r_p = np.dot(np.array(stock_split_i),w)
      #     portfolios_reg_info[port_type][j]['weights'][i_split].append(w)
      #     portfolios_reg_info[port_type][j]['cov_div'][i_split].append(div_ratio(w,S))
      #     portfolios_reg_info[port_type][j]['var_div'][i_split].append(var_div_ratio(w,np.array(stock_split_i)))
      #     portfolios_reg_info[port_type][j]['sharpe'][i_split].append(sharpe(mu_p, np.sqrt(var_p)))
      #     portfolios_reg_info[port_type][j]['omega'][i_split].append(omega(r_p))
      #     portfolios_reg_info[port_type][j]['sortino'][i_split].append(sortino(mu_p, r_p))
      #     beta_p = beta(r_p, snp500_i)
      #     portfolios_reg_info[port_type][j]['beta'][i_split].append(beta_p)
      #     portfolios_reg_info[port_type][j]['treynor'][i_split].append(treynor(mu_p, beta_p))


      
          
  dates = np.array(dates4)

  assert len(graph_dict[0]) == len(graph_dict[1]), f"{k} 0 and 1 not same"
  assert len(graph_dict[0]) == len(graph_dict[2]), f"{k} 0 and 2 not same"
  assert len(graph_dict[1]) == len(graph_dict[2]), f"{k} 1 and 2 not same"


  return {'dates':dates, 'graph_dict':graph_dict, 'sector':k, 'cov_dict':cov_dict, 'prec_dict':prec_dict, 'esg_mean':esg_mean, 'esg_std':esg_std,
  'esg_max':esg_max, 'esg_min':esg_min,
  'prec_dict_plus_1':prec_dict_plus_1, 'prec_dict_minus_1':prec_dict_minus_1, 'opt_lambda':opt_lambda, 'reg_lambda':reg_lambda,
  'return_dict':return_dict, 'window_size':window_size, 'stock_partition':stock_partition, 'portfolios_info':portfolios_info,
  'portfolios_reg_info':portfolios_reg_info}

if __name__ == '__main__':

  study = 'TEST'
  d = 1
  winow_len = 300
  graph_estimation = 'huge_glasso_ebic'

  scale = True
  transform = None#'nonparanormal'
  lamda = np.exp(np.linspace(start = np.log(1e-1), stop = np.log(1e-3), num = 50))

  print(graph_estimation)
  print(winow_len)
  print(scale)
  print(sector_classification.keys())


  # if not nested_i then the test will perform on each sector using sector_classfication dictionary already created
  asset_dict = dict()
  if study == 'sector':
    asset_dict = sector_classification
  elif study == 'nested_1':
    asset_dict[study] = np.concatenate((sector_classification['Utilities'], sector_classification['Energy'], sector_classification['Basic Materials']))
  elif study == 'TEST':
    asset_dict = {'Industrials':sector_classification['Industrials']}
  
  with Pool(1) as pool:
    L = pool.starmap(graph_est, [(price_pivot.loc[:, np.isin(price_pivot.columns,asset_dict[k])], 
                                  gp_esg_stock.loc[:, np.isin(gp_esg_stock.columns,asset_dict[k])], 
                                  asset_dict[k], 
                                    k, 
                                    d,
                                    winow_len, 
                                    3,
                                    graph_estimation,
                                    scale,
                                    transform,
                                    lamda) for k in asset_dict.keys()])#sector_classification.keys()

  with open(f'data/Graphs/{study}_d_{d}_winlen_{winow_len}_gest_{graph_estimation}_scale_{scale}_trans_{transform}.pkl', 'wb') as f:
    pickle.dump(L, f)




  # scale = False
  # transform = None
  # lamda = np.exp(np.linspace(start = np.log(1e-3), stop = np.log(1e-7), num = 100))

  # with Pool(4) as pool:
  #   L = pool.starmap(graph_est, [(price_pivot.loc[:, np.isin(price_pivot.columns,asset_dict[k])], 
  #                                 gp_esg_stock.loc[:, np.isin(gp_esg_stock.columns,asset_dict[k])], 
  #                                 asset_dict[k], 
  #                                   k, 
  #                                   d,
  #                                   winow_len, 
  #                                   3,
  #                                   graph_estimation,
  #                                   scale,
  #                                   transform,
  #                                   lamda) for k in asset_dict.keys()])#sector_classification.keys()

  # with open(f'data/Graphs/{study}_{d}_winlen_{winow_len}_gest_{graph_estimation}_scale_{scale}_trans_{transform}.pkl', 'wb') as f:
  #   pickle.dump(L, f)



  # scale = True
  # transform = None
  # lamda = np.exp(np.linspace(start = np.log(1e-1), stop = np.log(1e-4), num = 100))

  # with Pool(4) as pool:
  #   L = pool.starmap(graph_est, [(price_pivot.loc[:, np.isin(price_pivot.columns,asset_dict[k])], 
  #                                 gp_esg_stock.loc[:, np.isin(gp_esg_stock.columns,asset_dict[k])], 
  #                                 asset_dict[k], 
  #                                   k, 
  #                                   d,
  #                                   winow_len, 
  #                                   3,
  #                                   graph_estimation,
  #                                   scale,
  #                                   transform,
  #                                   lamda) for k in asset_dict.keys()])#sector_classification.keys()

  # with open(f'data/Graphs/{study}_{d}_winlen_{winow_len}_gest_{graph_estimation}_scale_{scale}_trans_{transform}.pkl', 'wb') as f:
  #   pickle.dump(L, f)

