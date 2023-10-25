import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
from sklearn.model_selection import GridSearchCV
from scipy.signal import find_peaks


def gmm_bic_score(estimator, X):
    """Callable to pass to GridSearchCV that will use the BIC score."""
    # Make it negative since GridSearchCV expects a score to maximize
    return -estimator.bic(X)


def model_selection(min_n, max_n):
    param_grid = {"n_components": range(min_n, max_n)}
    grid_search = GridSearchCV(
        GaussianMixture(), param_grid=param_grid, scoring=gmm_bic_score
    )
    return grid_search


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
    total_crystals = data['mu0'].values
    for i in data.columns:
        if 'pop_bin' not in i:
            data.drop(i, axis=1, inplace=True)
    # count the number of elements larger than 0 in each row
    # todo: it is not feasible since the non-zero elements are still around 500
    n_elements = [np.count_nonzero(data.iloc[i, :]) for i in range(data.shape[0])]
    # copy the data
    psd_data = data.copy(deep=True)
    # normalize psd to pdf
    for i in range(data.shape[0]):
        data.iloc[i, :] = data.iloc[i, :] / total_crystals[i]  # the probability

    # initialize the number of clusters
    if n_clusters is not None:
        gmm = GaussianMixture(n_components=n_clusters)
    else:
        # use the GridSearchCV to find the best n_clusters
        # gmm = model_selection(2, 12)
        # p0 = data.iloc[0, :].values * dL  # fit the data at time 0
        # d0 = np.random.choice(a=x, size=5 * round(total_crystals[0]), p=p0)
        # d0 = d0.reshape(-1, 1)
        # gmm.fit(d0)
        # n_clusters = gmm.best_params_['n_components']
        # print(n_clusters)
        # gmm = GaussianMixture(n_components=n_clusters)
        peaks_id, _ = find_peaks(data.iloc[0, :], height=0.001)
        n_clusters = len(peaks_id)
        gmm = GaussianMixture(n_components=n_clusters)

    # initialize the means and covariances
    peaks_id, _ = find_peaks(data.iloc[0, :], height=0.001)
    centers_init = x[peaks_id].reshape(-1, 1)
    covariances_init = 100 * np.ones((len(peaks_id), 1))
    weights_init = _['peak_heights']/np.sum(_['peak_heights'])
    p0 = data.iloc[0, :].values * dL  # fit the data at time 0
    d0 = np.random.choice(a=x, size=5 * round(total_crystals[0]), p=p0)
    d0 = d0.reshape(-1, 1)
    gm0 = GaussianMixture(n_components=len(peaks_id), means_init=centers_init,
                          weights_init=weights_init)
    gm0.fit(d0)
    mu0 = gm0.means_
    sigma0 = gm0.covariances_
    weights0 = gm0.weights_

    # read the data row by row and fit the data
    mu_all = []
    sigma_all = []
    weights_all = []
    precision_all = []
    n_rows = data.shape[0]
    for i in range(n_rows):
        # get the data(i.e. the pdf) of the row
        p = data.iloc[i, :].values * dL
        # sample x based on the pdf
        d = np.random.choice(a=x, size=5 * round(total_crystals[i]), p=p)
        # reshape the data
        d = d.reshape(-1, 1)
        # fit the data
        gmm.fit(d)
        # return the means, covariances and weights
        m, s, w = gmm.means_, gmm.covariances_, gmm.weights_
        precision = gmm.precisions_
        mu_all.append(m)
        sigma_all.append(s)
        weights_all.append(w)
        precision_all.append(precision)

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
        # todo: confused about pdf
        y_all = np.zeros((1, x.size))
        for i in range(len(peaks_id)):
            # y_all += weights_all[0][i] * norm.pdf(x, mu_all[0][i], sigma_all[0][i])
            y_all += weights0[i] * norm.pdf(x, mu0[i], sigma0[i])
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

    return mu_all, sigma_all, weights_all, precision_all


# todo: find the best n_clusters


if __name__ == '__main__':
    fp = 'data/'
    fn = 'PBEsolver_InputMat_231015_2234_runID1.csv'
    n = 8

    # load x
    L_max = 500  # [um]
    dL = 0.5  # [um]
    L_bounds = np.arange(0, L_max + dL, dL)  # [um]
    L_mid = np.mean([L_bounds[:-1], L_bounds[1:]], axis=0)  # [um]
    x = L_mid

    mu_ini, sigma_ini, weights_ini, precision_ini = split_gmm(fp, fn, x, dL, plot_fig=True)
    print(mu_ini)
