# -*- coding: utf-8 -*-
# @Time : 25/10/2023
# @Author : Yuequn Zhang
# @File : base_function.py
# @Project : GMM
import numpy as np
import pandas as pd
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


def error_func_scaled(y_true, y_pred):
    """
    calculate the error functions scaled by the sum of the true values
    :param y_true: the true values
    :param y_pred: the predicted values
    :return: the scaled root mean squared error
    """
    # calculate the error functions
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    s_rmse = rmse / np.sum(y_true)
    return s_rmse


def error_func_smape(y_true, y_pred):
    """
    calculate the Symmetric Mean Absolute Percentage Error
    :param y_true: the true values
    :param y_pred: the predicted values
    :return: the relative root mean squared error
    """
    if type(y_true) != np.ndarray:
        y_true = np.array(y_true).reshape(-1, 1)
    # numerator = np.abs(y_true - y_pred)
    # denominator = 0.5 * np.abs(y_true) + 0.5 * np.abs(y_pred)

    # note that if using the relative error, the result can be 1) a value between 0 and 1, 2) infinite, where the
    # numerator is non-zero and the denominator is zero, 3) nan, where both the numerator and denominator are zero
    # but for our smape, it can be divided into two cases: (a) the denominator is zero when the true value and the
    # prediction is zero at the same time; (b) the denominator is non-zero when either the true value or the
    # prediction is non-zero, which results in a value between 0 and 1
    # but for the first case, we can set the error to be zero since y_true = y_pred = 0
    # the other issue is for those very small y_trues, the denominator can be very small, which results in a very large

    # calculate the error
    error_vector = np.abs(y_true - y_pred) / (0.5 * np.abs(y_true) + 0.5 * np.abs(y_pred))
    # try to replace those true values that are very small (smaller than 0.01) with zero in the error
    error_vector[np.where(y_true < 0.1)] = 0
    # calculate the mean of the error
    error_smape = np.mean(error_vector) * 100

    # try to fill the nan with 0
    error_smape = np.nan_to_num(error_smape)
    return error_smape


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


def gm_best_model(data_bins_vec, x_vec, dL, total_crystals, err_threshold, n_component, optimize=False, max_component=30):
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
        while rmse > err_threshold and n_component < max_component:
            print('It needs to add clusters')
            n_component += 1
            # todo: if the deviation from threshold is too large, then add more components
            gmm = GaussianMixture(n_components=n_component)
            gmm.fit(d)
            m, s, w = gmm.means_, gmm.covariances_, gmm.weights_
            rmse = error_gmm(data_bins_vec, x_vec, m, s, w)

    return gmm, n_component


def psd_2_pdf(data):
    """
    transform the psd data into pdf
    :param data: the psd data
    :return: the pdf
    """
    # judge whether the data is a vector
    if len(data.shape) == 1:
        data = pd.DataFrame(data).T
    # calculate the sum of each row
    total_num = np.sum(data, axis=1)
    # convert the series into numpy array
    total_num = np.array(total_num).reshape(-1, 1)
    # calculate the pdf
    pdf = data / total_num
    # check whether the sum of each row is 1
    for i in range(pdf.shape[0]):
        if abs(sum(pdf.iloc[i, :]) - 1) > 0.0001:
            raise ValueError('The sum of each row is not 1')
    return total_num, pdf


def pdf_2_psd(pdf, total_number):
    """
    transform the pdf into psd
    :param pdf: the pdf of the whole data # todo: consider the case where the pdf is a vector
    :param total_number: the total number of crystals
    :return: the psd
    """
    # calculate the psd
    psd = pdf * total_number
    return psd


def cdf_func(pdf_data):
    """
    calculate the cdf of the data
    :param pdf_data: Dataframe, the probability density data
    :param x: vector, the x values
    :return: Dataframe, the cdf
    """
    # create a dataframe with the same shape as pdf_data
    cdf = pd.DataFrame(np.zeros(pdf_data.shape))
    # calculate the cumulative sum
    for i in range(pdf_data.shape[0]):
        cdf.iloc[i, :] = np.cumsum(pdf_data.iloc[i, :])
    return cdf


def reformat_input_output(input_mat, output_mat, n_ob_input, n_ob_output, t_sample_frac=0.25, no_sims=5000, shuffle=False):
    """
    Reformat input and output data for training
    :param input_mat: Dataframe, input matrix with observation info
    :param output_mat: dict, all simulation results with observation info
    :param n_ob_input: int, number of observations in input matrix
    :param n_ob_output: int, number of observations in output matrix
    :param t_sample_frac: Float, fraction of timepoints in each simulation to sample
    :param no_sims: number of simulations to use
    :param shuffle: bool, whether to shuffle dataset or not
    :return: Tuple, training data as input and output array
    """

    input_columns = ['runID', 'T0', 'dT', 'dt', 'S0', 'sol_k0', 'sol_kT', 'growth_k0', 'growth_kS',
                     'nuc_k0', 'nuc_kS', 'ini_mu0'] + [f"ob_{x}" for x in range(n_ob_input)] + ['width', 'middle']
    output_columns = ["c"] + [f"ob_{x}" for x in range(n_ob_output)] + ['width', 'middle']

    X, Y = [], []
    for runID, res in output_mat.items():
        res = res.sample(frac=t_sample_frac)

        no_timepoints = res.shape[0]
        Y.append(np.array(res[output_columns]))

        relevant_inputs = np.array(input_mat.query("runID == @runID")[input_columns])
        relevant_inputs_repeated = np.vstack([relevant_inputs] * no_timepoints)

        t_vec = np.array(res["t"])[..., np.newaxis]
        x = np.hstack([t_vec, relevant_inputs_repeated])

        X.append(x)
        if len(X) > no_sims:
            break

    X = np.vstack(X)
    Y = np.vstack(Y)

    if shuffle:
        ix = np.random.permutation(X.shape[0])
        X = X[ix, :]
        Y = Y[ix, :]

    print("X, Y dimensions: ", X.shape, Y.shape)

    return X, Y


if __name__ == '__main__':
    filepath = 'data/'
    file_name = 'PBEsolver_InputMat_231015_2234_runID1.csv'
    data = pd.read_csv(filepath + file_name)
    for i in data.columns:
        if 'pop_bin' not in i:
            data.drop(i, axis=1, inplace=True)
    # get the pdf
    total_number, pdf = psd_2_pdf(data)
    print(pdf)
    # get the cdf
    cdf = cdf_func(pdf)
    print(cdf)


