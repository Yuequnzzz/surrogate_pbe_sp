import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == '__main__':
    # load the data
    data_dist_est = pd.read_csv('distribution_est.csv', index_col=0)
    data_dist_est = data_dist_est.to_numpy()
    data_loc_est = pd.read_csv('location_est.csv', index_col=0)
    data_loc_est = data_loc_est.to_numpy()
    data_dist_test = pd.read_csv('distribution_test.csv', index_col=0)
    data_dist_test = data_dist_test.to_numpy()
    data_loc_test = pd.read_csv('location_test.csv', index_col=0)
    data_loc_test = data_loc_test.to_numpy()

    # Plot
    test_middle = data_loc_test[:, -1]
    test_width = data_loc_test[:, -2]
    n_ob_test = data_dist_test.shape[1]
    dL = 0.5

    # center around test_middle, generate 1/2*(n_ob_test-1) evenly spaced points, with interval test_width
    x_test_left = -test_width / 2 * (n_ob_test - 1) + test_middle * dL
    x_test_right = test_width / 2 * (n_ob_test - 1) + test_middle * dL
    x_test_loc = np.linspace(x_test_left, x_test_right, n_ob_test).T

    # get the observation locations for prediction
    pre_middle = data_loc_est[:, -1]
    pre_width = data_loc_est[:, -2]
    n_ob_pre = data_dist_est.shape[1]   # exclude c, width, middle
    x_pre_left = -pre_width / 2 * (n_ob_pre - 1) + pre_middle * dL
    x_pre_right = pre_width / 2 * (n_ob_pre - 1) + pre_middle * dL
    x_pre_loc = np.linspace(x_pre_left, x_pre_right, n_ob_pre).T

    # Plot the result
    # plot_id = random.sample(range(400), 10)
    plot_id = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    for p in plot_id:
        plt.figure(figsize=(8, 6))
        plt.plot(x_test_loc[p, :], data_dist_test[p, :], label="solver", color="b")
        plt.plot(x_pre_loc[p, :], data_dist_est[p, :], label="surrogate", color="g", ls="--")
        plt.legend(prop={'size': 25})

        plt.show()