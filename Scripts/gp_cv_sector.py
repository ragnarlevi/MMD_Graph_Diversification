import numpy as np
import pandas as pd


import scipy
import os

from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, DotProduct, ExpSineSquared, WhiteKernel
from sklearn.model_selection import RandomizedSearchCV, train_test_split, StratifiedKFold

from multiprocessing import Pool
import pickle


def fit_gp_cv(kernel, X_obs, y_obs, X_pred, scale = True):


    # Create Cross validation
    index = np.array(range(y_obs.shape[0]))
    cv = list()
    for i in [0,1,2,3]:
        test = index[i:len(index):4]
        train = index[~np.isin(index,test)]
        cv.append((train, test))

    n_iter = 120

    # rs = ShuffleSplit(n_splits=5, test_size=.25, random_state=0)
    distributions = dict(alpha=np.linspace(0.001, 0.3, n_iter))# dict(alpha=uniform(loc = 0, scale = 0.5))

    clf = RandomizedSearchCV(gaussian_process.GaussianProcessRegressor(kernel = kernel, n_restarts_optimizer=30, normalize_y= scale),
                                                                         distributions, random_state=0, cv = cv, n_iter  = n_iter)
    search = clf.fit(X_obs, y_obs)
    gp = gaussian_process.GaussianProcessRegressor(kernel=kernel,normalize_y = scale, n_restarts_optimizer=30, alpha =  search.best_params_['alpha']).fit(X_obs, y_obs)

    mean_prediction, std_prediction = gp.predict(X_pred, return_std=True)

    return mean_prediction, std_prediction,  gp, search

def fit_gp_envelope(i,esg_data):
    """
    Just a wrapper for fit_gp_cv above

    i: column number
    esg_data: data frame of esg_data
    
    """

    print(f'{i} {esg_data.shape[1]} \n')

    kernel = 1*Matern(length_scale=1, nu = 1.5)

    X = np.array(range(len(esg_data.index)))
    y = np.array(esg_data.iloc[:,i])
    obs_index = np.isfinite(y)

    y_obs = y[obs_index]
    X_obs = np.expand_dims(X[obs_index], axis = 1)
    X = np.expand_dims(X, axis=1)

    X_pred = np.expand_dims(np.array(range(X.shape[0])), axis = 1)
    mean_prediction, std_prediction,  gp, search = fit_gp_cv(kernel, X_obs, y_obs, X_pred)


    info = {'mean_prediction':mean_prediction,
    'std_prediction':std_prediction,
    'gp':gp,
    'search':search
    }


    name = esg_data.columns[i]

    out = {name:info}


    return out

if __name__ == '__main__':
    print(os.getcwd())

    # the sector index has no nan values as it has been filled with previous observalue in the creation of it
    # so sector_index_no_diff is a dict which contains the data and an index telling which observation where in fact observed
    esg_refined = pd.read_pickle('data/tidy/sector_index_no_diff.pkl')
    index = esg_refined['index']  # vector of bools containing where observation happened
    d = esg_refined['data']
    

    for k in d.columns:
        d[k].loc[~index[k]] = np.nan

    with Pool(6) as pool:
        L = pool.starmap(fit_gp_envelope, [(i, d) for i in list(range(d.shape[1]))])

    
    with open(f'data/tidy/gp_esg_sector.pkl', 'wb') as f:
            pickle.dump(L, f)
            

    