"""
Try the following regression models on the encoded dataset:
    -RandomForestRegressor
    -DecisionTree
"""
import os
import sys
import numpy as np
import pandas as pd
import time

from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, os.path.abspath('train_NN.py'))

from train_NN import calculate_errors


def regression_model_encoded(X, Y, model_name, hyperparameters, dL, test_ratio=0.1):
    """


    Returns:

    """
    # consider the unreliable runID and remove them
    unreliable_runID = [19, 42, 54, 101, 114, 149, 161, 163, 171, 190, 272, 273, 276]
    Y = Y[~np.isin(X[:, 1], unreliable_runID), :]
    X = X[~np.isin(X[:, 1], unreliable_runID), :]
    # find the number of unique runID
    n_runID = np.unique(X[:, 1]).shape[0]
    # based on test_ratio, split into training and testing data
    n_test_runID = int(n_runID * test_ratio)
    # randomly select n_test_runID runID
    np.random.seed(0)
    test_runID = np.random.choice(np.unique(X[:, 1]), n_test_runID, replace=False)
    # find the index of test_runID
    test_runID_index = np.where(np.isin(X[:, 1], test_runID))[0]
    # find the index of train_runID
    train_runID_index = np.where(~np.isin(X[:, 1], test_runID))[0]

    # remove column with run ID
    X_del = np.delete(X, 1, 1)

    # build the test-training set
    X_test = X_del[test_runID_index, :]
    X_train = X_del[train_runID_index, :]
    Y_test = Y[test_runID_index, :]
    Y_train = Y[train_runID_index, :]

    # set up the model
    if model_name == 'RandomForestRegressor':
        model_rg = RandomForestRegressor(**hyperparameters)
    elif model_name == '':
        pass

    # train the neural network
    train_start = time.time()
    model_rg.fit(X_train, Y_train)
    training_time = time.time() - train_start
    print("Training time: ", training_time)

    # predict the result
    test_start = time.time()
    y_pre = model_rg.predict(X_test)
    predict_time = time.time() - test_start
    print("Predict time: ", predict_time)

    # Calculate different error metrics
    rmse, c_rmse, mom0_err, mom0_err_rel, mom3_err, mom3_err_rel, av_len_rmse = calculate_errors(Y_test, y_pre)
    errors = {"RMSE_tot": rmse, "RMSE_c": c_rmse, "mu0": mom0_err, "mu0_rel": mom0_err_rel, "mu3": mom3_err, "mu3_rel": mom3_err_rel, "av_len": av_len_rmse}
    errors = pd.DataFrame(errors, index=[0])
    print(errors)

    test_middle = Y_test[:, -1]
    test_width = Y_test[:, -2]
    n_ob_test = Y_test.shape[1] - 3  # exclude c, width, middle
    # center around test_middle, generate 1/2*(n_ob_test-1) evenly spaced points, with interval test_width
    x_test_left = -test_width / 2 * (n_ob_test - 1) + test_middle * dL
    x_test_right = test_width / 2 * (n_ob_test - 1) + test_middle * dL
    x_test_loc = np.linspace(x_test_left, x_test_right, n_ob_test).T

    # get the observation locations for prediction
    pre_middle = y_pre[:, -1]
    pre_width = y_pre[:, -2]
    n_ob_pre = y_pre.shape[1] - 3  # exclude c, width, middle
    x_pre_left = -pre_width / 2 * (n_ob_pre - 1) + pre_middle * dL
    x_pre_right = pre_width / 2 * (n_ob_pre - 1) + pre_middle * dL
    x_pre_loc = np.linspace(x_pre_left, x_pre_right, n_ob_pre).T

    # Plot the result
    plt.figure(figsize=(6, 8))
    # in the title, include the number of nodes per layer and the number of layers
    plt.suptitle(f"{hyperparameters}")
    # the distribution
    plt.subplot(3, 1, 1)
    plt.xlim((0, 500))
    plt.plot(x_test_loc[10, :], Y_test[10, 1:-2], label="case_one", color="b")
    plt.plot(x_pre_loc[10, :], y_pre[10, 1:-2], label="case_one_pred", color="g")
    plt.legend()
    plt.ylabel(r"PSD $f$ [m$^{-3}\mu$m$^{-1}$]")

    plt.subplot(3, 1, 2)
    plt.xlim((0, 500))
    plt.plot(x_test_loc[-3, :], Y_test[-3, 1:-2], label="case_two", color="r")
    plt.plot(x_pre_loc[-3, :], y_pre[-3, 1:-2], label="case_two_pred", color="g")
    plt.legend()

    # the concentration through the whole process
    plt.subplot(3, 1, 3)
    plt.scatter(X_test[0, 0], Y_test[0, 0], label='true_concentration', color='r')
    plt.scatter(X_test[0, 0], y_pre[0, 0], label='pred_concentration', color='g')
    plt.xlabel(r'')
    plt.ylabel(r"Concentration")
    plt.legend()

    # save the plot
    # plt.savefig(f"data/Prediction_hyperparameter/fig/Encoded_figure_{dt.now().strftime('%y%m%d_%H%M')}_{nodes_per_layer}_{no_layers}.png")
    # show the plot
    plt.show()

    return


if __name__ == '__main__':

    # tune some parameters
    encoded = True
    dL = 0.5
    test_ratio = 0.1

    # save_name = "InputMat_231108_1637"  # small case (10) with only temperature varying, gaussian like
    # save_name = "InputMat_231110_1058"  # small case (10) with parameters regarding growth and nucleation constant
    save_name = "InputMat_231111_2012"  # large case (300) with parameters regarding growth and nucleation constant
    # save_name = "InputMat_231109_1543"  # small case (10) with all varied
    # save_name = "InputMat_231021_0805"  # large case (200)
    # save_name = "InputMat_231110_0720"  # large case (300)

    import_file_input = f'D:/PycharmProjects/GMM/data/sparse_training_data/{save_name}_input_61_61.csv'
    import_file_output = f'D:/PycharmProjects/GMM/data/sparse_training_data/{save_name}_output_61_61.csv'
    X = pd.read_csv(import_file_input, index_col=0)
    Y = pd.read_csv(import_file_output, index_col=0)
    # convert to numpy array
    X = X.to_numpy()
    Y = Y.to_numpy()

    regression_model_encoded(X=X, Y=Y, model_name='RandomForestRegressor',
                             hyperparameters={'max_depth': 20, 'n_estimators': 50},
                             dL=dL, test_ratio=test_ratio)



