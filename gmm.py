import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
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


def split_gmm(filepath, file_name, n_clusters, x, dL, plot_fig=False):
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
    total_crystals = data['mu0']
    for i in data.columns:
        if 'pop_bin' not in i:
            data.drop(i, axis=1, inplace=True)
    # copy the data
    psd_data = data.copy(deep=True)
    # check whether the data to be fitted is pdf, if not, normalize it
    integral = [np.nansum(data.iloc[i, :] * dL) for i in range(data.shape[0])]
    for i in range(data.shape[0]):
        data.iloc[i, :] = data.iloc[i, :] / integral[i] * dL  # the probability
    total = [np.sum(data.iloc[i, :]) for i in range(data.shape[0])]

    # create a GMM
    gmm = GaussianMixture(n_components=n_clusters)
    # read the data row by row and fit the data
    mu_all = []
    sigma_all = []
    weights_all = []
    n_rows = data.shape[0]
    for i in range(n_rows):
        # get the data(i.e. the pdf) of the row
        p = data.iloc[i, :].values
        # sample x based on the pdf
        d = np.random.choice(a=x, size=5 * round(integral[i]), p=p)
        # # calculate the frequency of each bin
        # frequency = psd_data.iloc[i, :] * dL
        # d = []
        # for j in range(len(frequency)):
        #     d += [x[j]] * round(frequency[j])  # todo: have some round errors here
        # d = np.array(d)
        # # plot to check whether the sampling makes sense
        # plt.bar(x, frequency, color='b', alpha=0.5)
        # plt.show()
        # reshape the data
        d = d.reshape(-1, 1)
        # fit the data
        gmm.fit(d)
        # return the means, covariances and weights
        m, s, w = gmm.means_, gmm.covariances_, gmm.weights_
        mu_all.append(m)
        sigma_all.append(s)
        weights_all.append(w)

    # plot the data
    if plot_fig:
        plt.figure(figsize=(6, 8))

        plt.subplot(3, 1, 1)
        plt.plot(x, 1 / dL * data.iloc[0, :], label="start", color="b")
        plt.plot(x, 1 / dL * data.iloc[-1, :], label="end", color="r", ls="--")
        plt.legend()
        plt.ylabel(r"PSD $f$ [m$^{-3}\mu$m$^{-1}$]")

        plt.subplot(3, 1, 2)
        plt.plot(x, 1 / dL * data.iloc[0, :], label="start_ori", color="b")
        # todo: confused about pdf
        y_all = np.zeros((1, x.size))
        for i in range(n_clusters):
            # y = weights_all[0][i] * norm.pdf(x, mu_all[0][i], sigma_all[0][i])
            # plt.plot(x, y.reshape(-1, 1), color="g", ls="--")
            y_all += weights_all[0][i] * norm.pdf(x, mu_all[0][i], sigma_all[0][i])
        plt.plot(x, y_all.reshape(-1, 1), label="estimate", color="g", ls="--")
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.plot(x, 1 / dL * data.iloc[-1, :], label="end_ori", color="r")
        y2 = np.zeros((1, x.size))
        for i in range(n_clusters):
            y2 += weights_all[-1][i] * norm.pdf(x, mu_all[-1][i], sigma_all[-1][i])
        plt.plot(x, y2.reshape(-1, 1), label="estimate", color="g", ls="--")
        plt.legend()
        plt.xlabel(r"size $L$ [um]")
        plt.show()

    return mu_all, sigma_all, weights_all


# todo: find the best n_clusters


if __name__ == '__main__':
    fp = 'data/'
    fn = 'PBEsolver_InputMat_231015_2234_runID0.csv'
    n = 12

    # load x
    L_max = 500  # [um]
    dL = 0.5  # [um]
    L_bounds = np.arange(0, L_max + dL, dL)  # [um]
    L_mid = np.mean([L_bounds[:-1], L_bounds[1:]], axis=0)  # [um]
    x = L_mid

    # model_selection(3, 10).fit(x.reshape(-1, 1))
    mu, sigma, weights = split_gmm(fp, fn, n, x, dL, plot_fig=True)
