# -*- coding: utf-8 -*-
# @Time : 25/10/2023
# @Author : Yuequn Zhang
# @File : base_function.py
# @Project : GMM
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV


def gmm_bic_score(estimator, X):
    """Callable to pass to GridSearchCV that will use the BIC score."""
    # Make it negative since GridSearchCV expects a score to maximize
    return -estimator.bic(X)


def model_selection(min_n, max_n):
    param_grid = {"n_components": range(min_n, max_n), "covariance_type": ["spherical", "tied", "diag", "full"]}
    grid_search = GridSearchCV(
        GaussianMixture(), param_grid=param_grid, scoring=gmm_bic_score
    )
    return grid_search


def error_functions(y_true, y_pred):
    """
    calculate the error functions
    :param y_true: the true values
    :param y_pred: the predicted values
    :return: the root mean squared error
    """
    # calculate the error functions
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    return rmse


def error_gmm(y_true, x, mu, sigma, weights):
    """
    calculate the error functions of the gmm
    :param y_true: the true values
    :param x: the x values for the pdf
    :param mu: the means of a certain line
    :param sigma: the covariances
    :param weights: the weights
    :return: the errors
    """
    n = len(mu)
    y_pred = np.zeros((1, x.size))
    for i in range(n):
        y_pred += weights[i] * norm.pdf(x, mu[i], sigma[i])
    rmse = error_functions(y_true, y_pred.reshape(-1, 1))
    return rmse


def gm_best_model(data_bins_vec, x_vec, dL, total_crystals, err_threshold, n_component, optimize=False):
    """
    find the best n_component for gmm
    :param data_bins_vec: the data of bin values in the form of (, n_features), where the sample is transformed into pdf
    :param x_vec: the vector x
    :param dL: the width of each bin
    :param total_crystals: the total number of crystals
    :param err_threshold: the threshold of the error
    :param n_component: the initial number of components
    :param optimize: control whether to optimize the number of components
    :return: the best model and the number of components
    """
    # initial trained model
    gmm = GaussianMixture(n_components=n_component)
    # get the probability
    p = data_bins_vec * dL
    # sample x based on the probability
    d = np.random.choice(a=x_vec, size=5 * round(total_crystals), p=p)
    # reshape the data
    d = d.reshape(-1, 1)
    # fit the data
    gmm.fit(d)

    if optimize:
        # return the means, covariances and weights
        m, s, w = gmm.means_, gmm.covariances_, gmm.weights_
        # calculate the errors
        rmse = error_gmm(data_bins_vec, x_vec, m, s, w)
        print(f"rmse = {rmse}")
        while rmse > err_threshold:
            print('It needs to add clusters')
            n_component += 1
            # todo: if the deviation from threshold is too large, then add more components
            gmm = GaussianMixture(n_components=n_component)
            gmm.fit(d)
            m, s, w = gmm.means_, gmm.covariances_, gmm.weights_
            rmse = error_gmm(data_bins_vec, x_vec, m, s, w)

    return gmm, n_component


