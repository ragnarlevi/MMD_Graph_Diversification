import numpy as np
import pandas as pd
import datetime
import os, sys
import warnings
import tqdm
import random
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
huge = importr('huge')


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
    gc()
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



def create_G(A:np.array, returns = None):
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
  if returns is not None:
    nx.set_node_attributes(G, {key: [value] for key, value in enumerate(returns)}, "returns")
  return G


def div_ratio(w, cov):
  # numerator is perfect correlation
  # denom is portfolio risk
  return np.inner(w, np.sqrt(np.diag(cov)))/np.sqrt(np.dot(w, cov).dot(w))

def var_div_ratio(w,data, q = 0.95):
  # w weights
  # d = data

  ind_var = np.zeros(data.shape[1])
  for col in range(data.shape[1]):
      ind_var[col] = np.quantile(-data[:,col], q)

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
  x = np.cumprod(np.exp(X_port))  # convert return to price
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

def graph_est(price_df, esg_df, all_stocks_in_sector, k,d = 1, window_size = 150, nr_splits = 3, graph_estimation = "lgmrf_heavy", scale = False, transform = None, lamda = [1, 0.1], ebic_gamma = 0.01, order_esg = True):
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


    # determine which stocks in the sector will be considered.
    # If the total number of stocks in the sector are not divisible by nr_splits
    # then the first stocks which correspond to the reminder of the division will be omitted
    # in future iterations only the stocks in stocks_considered will be used
  stocks_in_sector = price_df.columns[np.isin(price_df.columns,all_stocks_in_sector)]
  esg_stocks_in_sector = esg_df.columns[np.isin(esg_df.columns,all_stocks_in_sector)]

  stocks_considered[k] = np.array(stocks_in_sector.intersection(esg_stocks_in_sector))
  print(f'{k} has {len(stocks_considered[k])} stocks')
  

  if len(stocks_considered[k]) % nr_splits != 0:
    res = len(stocks_considered[k]) % nr_splits
    print(f'{k} dropped {stocks_considered[k][:res]}')
    stocks_considered[k] = np.array(stocks_considered[k][res:])

  # Find min maximum date
  # Find mi
  min_date = np.min((np.max(price_df.index), np.max(esg_df.index)))
  price_df = price_df.loc[price_df.index <= min_date]
  esg_df = esg_df.loc[esg_df.index <= min_date]


  the_range = range(window_size+100 , price_df.shape[0], 2)# price_df.shape[0]
  nr_asset = int(len(stocks_considered[k])/3)
  nr_its = len(the_range)
      
  graph_dict = {i: [nx.Graph() for _ in range(nr_its) ] for i in range(nr_splits)}
  graph_dict2 = {i: [] for i in range(nr_splits)}
  return_dict  = {i: np.zeros((nr_its,nr_asset)) for i in range(nr_splits)}
  esg_mean = {i: np.zeros(nr_its) for i in range(nr_splits)}
  esg_std = {i: np.zeros(nr_its) for i in range(nr_splits)}
  esg_max = {i: np.zeros(nr_its) for i in range(nr_splits)}
  esg_min = {i: np.zeros(nr_its) for i in range(nr_splits)}
  cov_dict = {i: np.zeros((nr_its,nr_asset,nr_asset)) for i in range(nr_splits)}
  prec_dict = {i: np.zeros((nr_its,nr_asset,nr_asset)) for i in range(nr_splits)}
  prec_dict_plus_1 = {i: np.zeros((nr_its,nr_asset,nr_asset)) for i in range(nr_splits)}
  prec_dict_minus_1 = {i: np.zeros((nr_its,nr_asset,nr_asset)) for i in range(nr_splits)}
  opt_lambda = {i: np.zeros((nr_its)) for i in range(nr_splits)}
  reg_lambda = {i: [] for i in range(nr_splits)}
  graph_covdict = {i: [] for i in range(nr_splits)}
  where_opt_dict = {i: np.zeros((nr_its)) for i in range(nr_splits)}
  where_opt_lambda_dict = {i: np.zeros((nr_its)) for i in range(nr_splits)}

  portfolios_info = {}
  portfolios_reg_info = {}

  for port_type in ['uniform', 'sharpe', 'gmv']:
    portfolios_info[port_type] = {}
    portfolios_info[port_type]['weights'] = {i: np.zeros((nr_its,nr_asset)) for i in range(nr_splits)}
    portfolios_info[port_type]['cov_div'] = {i: np.zeros((nr_its)) for i in range(nr_splits)}
    portfolios_info[port_type]['var_div'] = {i: np.zeros((nr_its)) for i in range(nr_splits)}
    portfolios_info[port_type]['omega'] = {i: np.zeros((nr_its)) for i in range(nr_splits)}
    portfolios_info[port_type]['sharpe'] = {i: np.zeros((nr_its)) for i in range(nr_splits)}
    portfolios_info[port_type]['sortino'] = {i: np.zeros((nr_its)) for i in range(nr_splits)}
    portfolios_info[port_type]['beta'] = {i: np.zeros((nr_its)) for i in range(nr_splits)}
    portfolios_info[port_type]['treynor'] = {i: np.zeros((nr_its)) for i in range(nr_splits)}
    portfolios_info[port_type]['max_draw'] = {i: np.zeros((nr_its)) for i in range(nr_splits)}

  nlambda = len(lamda)
  reg_interval = np.concatenate((np.arange(0, nlambda, 10), [nlambda-1]))

  stock_partition = {i: [['']*nr_asset for _ in range(nr_its)] for i in range(nr_splits)}
  

  min_max_date = np.min([np.max(price_df.index), np.max(esg_df.index)])
  price_df=price_df.loc[price_df.index <= min_max_date]
  i_cnt = 0

  for i in tqdm.tqdm(the_range): #price_df.shape[0]
    #print(f"Size of graph dict {len(graph_dict[0])}")

    if order_esg:
      # get esg scores of the stocks for the current iteration
      esg_i = np.array(esg_df[stocks_considered[k]].loc[esg_df.index == price_df.index[i]].iloc[0,:])
      # order stocks
      stocks_ordered_i = np.array(stocks_considered[k][np.argsort(esg_i)])
    else:
      random.shuffle(stocks_considered[k])
      stocks_ordered_i = np.array(stocks_considered[k])

    

    # get snp500 index for current iteration, used in  
    snp500_i = np.array(sp500.loc[np.isin(sp500.date,price_df.iloc[(i-window_size):i].index), 'log_return'])
    # get date of the iteration i
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

      print(f"stock_split_i shape {stock_split_i.shape}")
      
      # get standar deviation of each asset
      var = np.diag(np.cov(stock_split_i.T))
      if i == window_size:
        print(var.shape)

    

      esg_split_i_mean = esg_df[stocks_ordered_i[stocks_index]].loc[esg_df.index == price_df.index[i]].mean(axis = 1)[0]
      esg_mean[i_split][i_cnt] = esg_split_i_mean
      esg_split_i_std = esg_df[stocks_ordered_i[stocks_index]].loc[esg_df.index == price_df.index[i]].std(axis = 1)[0]
      esg_std[i_split][i_cnt] = esg_split_i_std
      esg_split_i_max = esg_df[stocks_ordered_i[stocks_index]].loc[esg_df.index == price_df.index[i]].max(axis = 1)[0]
      esg_max[i_split][i_cnt] = esg_split_i_max
      esg_split_i_min = esg_df[stocks_ordered_i[stocks_index]].loc[esg_df.index == price_df.index[i]].min(axis = 1)[0]
      esg_min[i_split][i_cnt] = esg_split_i_min

      stock_partition[i_split][i_cnt] = stocks_ordered_i[stocks_index]

      if i == window_size and i_split == 0:
        print(f"First iteration of {k}. Shape is {stock_split_i.shape}")

      X = np.array(stock_split_i)
      
      
      if graph_estimation == 'huge_glasso_ebic':

        out_select = my_huge(X, gamma = ebic_gamma, lamda = lamda, scale = scale)
        out_select = dict(zip(out_select.names, list(out_select)))
        where_optimal = int(np.where(out_select['opt.lambda'][0] == out_select['lambda'])[0][0])
        if (where_optimal == nlambda-1) or (where_optimal == 0):
          warnings.warn(f"Warning optimal graph is last or first regularization parameter, namely param no. {where_optimal} {out_select['lambda'][where_optimal]}...Will extend grid \n")
          if where_optimal == nlambda-1:
            # need to lower regularization
            lamda_tmp = np.exp(np.linspace(start = np.log(lamda[-1])+1, stop = np.log(lamda[-1]) -5, num = 150))
          else:
            lamda_tmp = np.exp(np.linspace(start = np.log(lamda[0])+3, stop = np.log(lamda[0])-1, num = 150))

          out_select = my_huge(X, gamma = ebic_gamma, lamda = lamda_tmp, scale = scale)
          out_select = dict(zip(out_select.names, list(out_select)))
          where_optimal = int(np.where(out_select['opt.lambda'][0] == out_select['lambda'])[0][0])
        
        
        where_opt_dict[i_split][i_cnt]= where_optimal
        where_opt_lambda_dict[i_split][i_cnt] = out_select['lambda'][where_optimal]


        precision_matrix = out_select['opt.icov'].copy()
        precision_matrix_no_diag = precision_matrix.copy()
        np.fill_diagonal(precision_matrix_no_diag,0)
        mu = np.mean(stock_split_i, axis=0)
        G = create_G(-precision_matrix_no_diag, returns=mu)
        graph_dict[i_split][i_cnt] = G


      #scaler = StandardScaler()
      if graph_estimation == 'lgmrf_heavy' or graph_estimation == 'lgmrf_normal':
        precision_matrix = precision_matrix + 0.001*np.identity(precision_matrix.shape[0])
      
      opt_lambda[i_split][i_cnt] = out_select['opt.lambda'][0]
      # reg_lambda[i_split].append(out_select['lambda'][reg_interval])
      # Calculate diversification using graph estimate
      S = np.linalg.inv(precision_matrix)
      # If X was scaled during graph construction
      # we need to get the covariance back (we do not want to use correlation for portfolio optimization)
      if scale:
        S = np.dot(np.diag(np.sqrt(var)), S).dot(np.diag(np.sqrt(var)))
      cov_dict[i_split][i_cnt] = S
      mu = np.mean(stock_split_i, axis=0)
      return_dict[i_split][i_cnt] = mu
      prec_dict[i_split][i_cnt] = precision_matrix
      if (where_optimal+1)<=(nlambda-1):
        prec_dict_plus_1[i_split][i_cnt] = out_select['icov'][where_optimal+1]
      if (where_optimal-1)>=0:
        prec_dict_minus_1[i_split][i_cnt] = out_select['icov'][where_optimal-1]

      for port_type in ['uniform', 'sharpe', 'gmv']:
        w, mu_p, var_p = portfolio(S, np.linalg.inv(S), mu, np.array(stock_split_i), port_type)
        r_p = np.dot(np.array(stock_split_i),w)
        # print(np.sum(r_p <0))
        portfolios_info[port_type]['weights'][i_split][i_cnt] = w
        portfolios_info[port_type]['cov_div'][i_split][i_cnt] = div_ratio(w,S)
        portfolios_info[port_type]['var_div'][i_split][i_cnt] = var_div_ratio(w,np.array(stock_split_i))
        portfolios_info[port_type]['sharpe'][i_split][i_cnt] = sharpe(mu_p, np.sqrt(var_p))
        portfolios_info[port_type]['omega'][i_split][i_cnt] = omega(r_p)
        portfolios_info[port_type]['sortino'][i_split][i_cnt] = sortino(mu_p, r_p)

        beta_p = beta(r_p, snp500_i)
        portfolios_info[port_type]['beta'][i_split][i_cnt] = beta_p
        portfolios_info[port_type]['treynor'][i_split][i_cnt] = treynor(mu_p, beta_p)
        portfolios_info[port_type]['max_draw'][i_split][i_cnt] = max_drawdown(r_p)

    

      for j in reg_interval:
        j = int(j)
        for port_type in ['uniform', 'sharpe', 'gmv']:
          S = np.linalg.inv(out_select['icov'][j])
          if scale:
            S = np.dot(np.diag(var), S).dot(np.diag(var))
          w, mu_p, var_p = portfolio(S, np.linalg.inv(S), mu, np.array(stock_split_i), port_type)
          r_p = np.dot(np.array(stock_split_i),w)
          portfolios_reg_info[port_type][j]['weights'][i_split].append(w)
          portfolios_reg_info[port_type][j]['cov_div'][i_split].append(div_ratio(w,S))
          portfolios_reg_info[port_type][j]['var_div'][i_split].append(var_div_ratio(w,np.array(stock_split_i)))
          portfolios_reg_info[port_type][j]['sharpe'][i_split].append(sharpe(mu_p, np.sqrt(var_p)))
          portfolios_reg_info[port_type][j]['omega'][i_split].append(omega(r_p))
          portfolios_reg_info[port_type][j]['sortino'][i_split].append(sortino(mu_p, r_p))
          beta_p = beta(r_p, snp500_i)
          portfolios_reg_info[port_type][j]['beta'][i_split].append(beta_p)
          portfolios_reg_info[port_type][j]['treynor'][i_split].append(treynor(mu_p, beta_p))
          portfolios_reg_info[port_type][j]['max_draw'][i_split].append(max_drawdown(r_p))

    print(robjects.r.gc())
    del out_select
    i_cnt +=1
            

  dates = np.array(price_df.index[the_range])

  assert len(graph_dict[0]) == len(graph_dict[1]), f"{k} 0 and 1 not same"
  assert len(graph_dict[0]) == len(graph_dict[2]), f"{k} 0 and 2 not same"
  assert len(graph_dict[1]) == len(graph_dict[2]), f"{k} 1 and 2 not same"




  data = {'dates':dates, 'graph_dict':graph_dict, 'graph_dict2':graph_dict2, 'sector':k, 'cov_dict':cov_dict, 'prec_dict':prec_dict, 'esg_mean':esg_mean, 'esg_std':esg_std,
      'esg_max':esg_max, 'esg_min':esg_min,'transform':transform,'scale':scale,
      'prec_dict_plus_1':prec_dict_plus_1, 'prec_dict_minus_1':prec_dict_minus_1, 'opt_lambda':opt_lambda, 'reg_lambda':reg_lambda,
      'return_dict':return_dict, 'window_size':window_size, 'stock_partition':stock_partition, 'portfolios_info':portfolios_info,
      'portfolios_reg_info':portfolios_reg_info, 'graph_covdict':graph_covdict, 'where_opt':where_opt_dict}

  with open(f'data/Graphs/{k}_{order_esg}_d_{d}_winlen_{window_size}_gest_{graph_estimation}_scale_{scale}_trans_{transform}.pkl', 'wb') as f:
        pickle.dump(data, f)   

  return None


if __name__ == '__main__':

  def spit_assets(study, sector_classification):
    assets = dict()
    if study == 'nested_1':
      assets = np.concatenate((sector_classification['Utilities'], sector_classification['Energy'], sector_classification['Basic Materials']))
    elif study == 'all':
      assets = sum(sector_classification.values(), [])
    else:
      assets = sector_classification[study]
    return assets

  d = 1
  winow_len = 300
  graph_estimation = 'huge_glasso_ebic'



  print(graph_estimation)
  print(winow_len)
  print(sector_classification.keys())




  graph_est(price_pivot.loc[:, np.isin(price_pivot.columns,spit_assets('Industrials', sector_classification))],
            gp_esg_stock.loc[:, np.isin(gp_esg_stock.columns,spit_assets('Industrials', sector_classification))],
            spit_assets('all', sector_classification),
            'all',
            d,
            winow_len,
            3,
            graph_estimation,
            False,
            'nonparanormal',
            np.exp(np.linspace(start = np.log(1e-3), stop = np.log(50e-5), num = 50)),
            0.1,
            True
            )

