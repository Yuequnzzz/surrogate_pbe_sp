import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
from scipy.signal import find_peaks

from base_functions import *


def split_gmm(filepath, file_name, x, dL, plot_fig=False, n_clusters=None):
    """
    split the data into n_clusters by GMM
    :param filepath: the path of the data
    :param file_name: the name of the csv file
    :param n_clusters: the number of clusters
    :param x: the x-axis
    :param plot_fig: whether to plot the data
    :return: the means and covariances of the data
    """
    # load data from csv file
    data = pd.read_csv(filepath + file_name)
    # determine whether the data is input matrix or output matrix
    if 'PBEsolver' in file_name:
        output_file = True
        print('We are now dealing with the output matrix')
    else:
        output_file = False
        print('We are now dealing with the input matrix')

    # get the total number of crystals
    if 'mu0' not in data.columns:
        total_crystals = data['ini_mu0'].values
    else:
        total_crystals = data['mu0'].values

    # drop the columns that are not related to the population
    for i in data.columns:
        if 'pop_bin' not in i:
            data.drop(i, axis=1, inplace=True)
    # copy the data
    psd_data = data.copy(deep=True)
    # normalize psd to pdf
    for i in range(data.shape[0]):
        data.iloc[i, :] = data.iloc[i, :] / total_crystals[i]  # the probability

    # initialize the number of clusters
    if n_clusters is not None:
        n_clusters = n_clusters
    else:
        # get the peaks of the pdf at time 0 to initialize the number of clusters
        # todo: if the distribution fluctuates too much, the number of clusters will be too large
        peaks_id, _ = find_peaks(data.iloc[0, :], height=0.001)
        n_clusters = len(peaks_id)

    if output_file:
        # read the data row by row and fit the data
        mu_all = []
        sigma_all = []
        weights_all = []
        n_possible = []
        n_rows = data.shape[0]
        gmm_all = []
        # find the best n_clusters for each row
        for i in range(n_rows):
            gmm, n = gm_best_model(
                data.iloc[i, :],
                x,
                dL,
                total_crystals[i],
                err_threshold=0.001,
                n_component=n_clusters,
                optimize=True
            )
            print('the best n_component for row %d is %d' % (i, n))
            n_possible.append(n)
            gmm_all.append(gmm)
        # get the maximum n_clusters
        n_clusters = max(n_possible)
        print('the maximum n_component is %d' % n_clusters)
        # find all the index where the value does not equal maximum n_clusters
        id_non_max = [i for i, k in enumerate(n_possible) if k != n_clusters]
        # fit the data again if the number of clusters is not the maximum
        for i in id_non_max:
            gmm, n = gm_best_model(
                data.iloc[i, :],
                x,
                dL,
                total_crystals[i],
                err_threshold=0.001,
                n_component=n_clusters,
                optimize=False
            )
            gmm_all[i] = gmm
            n_possible[i] = n

        # get the means, covariances and weights
        for i in range(n_rows):
            m = gmm_all[i].means_
            s = gmm_all[i].covariances_
            w = gmm_all[i].weights_
            mu_all.append(m)
            sigma_all.append(s)
            weights_all.append(w)

        # plot the data
        if plot_fig:
            plt.figure(figsize=(6, 8))

            plt.subplot(3, 1, 1)
            plt.plot(x, data.iloc[0, :], label="start", color="b")
            plt.plot(x, data.iloc[-1, :], label="end", color="r", ls="--")
            plt.legend()
            plt.ylabel(r"PSD $f$ [m$^{-3}\mu$m$^{-1}$]")

            plt.subplot(3, 1, 2)
            plt.plot(x, data.iloc[0, :], label="start_ori", color="b")
            y_all = np.zeros((1, x.size))
            for i in range(n_clusters):
                y_all += weights_all[0][i] * norm.pdf(x, mu_all[0][i], sigma_all[0][i])
            plt.plot(x, y_all.reshape(-1, 1), label="estimate", color="g", ls="--")
            plt.legend()

            plt.subplot(3, 1, 3)
            plt.plot(x, data.iloc[-1, :], label="end_ori", color="r")
            y2 = np.zeros((1, x.size))
            for i in range(n_clusters):
                y2 += weights_all[-1][i] * norm.pdf(x, mu_all[-1][i], sigma_all[-1][i])
            plt.plot(x, y2.reshape(-1, 1), label="estimate", color="g", ls="--")
            plt.legend()
            plt.xlabel(r"size $L$ [um]")
            plt.show()

        return mu_all, sigma_all, weights_all

    else: # file of input matrix
        # read the data row by row and fit the data
        mu_all = []
        sigma_all = []
        weights_all = []
        n_rows = data.shape[0]
        # find the best n_clusters for each row
        for i in range(n_rows):
            gmm, n = gm_best_model(
                data.iloc[i, :],
                x,
                dL,
                total_crystals[i],
                err_threshold=0.001,
                n_component=n_clusters,
                optimize=True
            )
            print('the best n_component for row %d is %d' % (i, n))
            m = gmm.means_
            print('the length of means is', len(m))
            s = gmm.covariances_
            w = gmm.weights_
            mu_all.append(m)
            sigma_all.append(s)
            weights_all.append(w)

        return mu_all, sigma_all, weights_all


if __name__ == '__main__':
    fp = 'data/'
    # fn = 'PBEsolver_InputMat_231015_2234_runID1.csv'
    fn = 'InputMat_231015_2234.csv'
    export_path = 'data/gmm_results/'
    n = 8

    # load x
    L_max = 500  # [um]
    dL = 0.5  # [um]
    L_bounds = np.arange(0, L_max + dL, dL)  # [um]
    L_mid = np.mean([L_bounds[:-1], L_bounds[1:]], axis=0)  # [um]
    x = L_mid

    mu, sigma, weights = split_gmm(fp, fn, x, dL, plot_fig=True, n_clusters=12)

    # save the data
    # create a empty dataframe
    cols = ['mu' + str(j) for j in range(len(mu[0]))] + ['sigma' + str(j) for j in range(len(sigma[0]))] + \
           ['weights' + str(j) for j in range(len(weights[0]))]
    df_gmm = pd.DataFrame(columns=cols)

    if 'PBEsolver' in fn:
        for i in range(len(mu)):
            df = pd.DataFrame(np.concatenate((mu[i].reshape(1, -1), sigma[i].reshape(1, -1), weights[i].reshape(1, -1)),
                                             axis=1), columns=cols)
            df_gmm = pd.concat([df_gmm, df], ignore_index=True)
        # save the dataframes into csv files
        df_gmm.to_csv(export_path + 'gmm_' + fn, index=False)
    else:
        for i in range(len(mu)):
            cols = ['mu' + str(j) for j in range(len(mu[i]))] + ['sigma' + str(j) for j in range(len(sigma[i]))] + \
                   ['weights' + str(j) for j in range(len(weights[i]))]
            df = pd.DataFrame(np.concatenate((mu[i].reshape(1, -1), sigma[i].reshape(1, -1), weights[i].reshape(1, -1)),
                                             axis=1), columns=cols)

            # save the dataframes into csv files
            df.to_csv(export_path + 'gmm_' + fn + '_' + str(i), index=False)
