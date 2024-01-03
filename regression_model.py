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
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.abspath('train_NN.py'))
from train_NN import calculate_errors


def train_test_regression(X, Y, model_name, hyperparameters, kFoldFlag=False, n_splits=5):
    # consider the unreliable runID and remove them
    unreliable_runID = []  # TODO: MODIFY
    Y = Y[~np.isin(X[:, 1], unreliable_runID), :]
    X = X[~np.isin(X[:, 1], unreliable_runID), :]
    X_del = np.delete(X, 1, 1)

    kf = KFold(n_splits=n_splits)

    training_time = []
    predict_time = []
    errors = {"RMSE_tot":[], "RMSE_c":[], "mu0":[], "mu0_rel":[], "mu3":[], "mu3_rel":[], "av_len":[]}
    y_ests = []
    for k, (train_ix, test_ix) in enumerate(kf.split(X_del)):
        # set up the model
        if model_name == 'RandomForestRegressor':
            model_rg = RandomForestRegressor(**hyperparameters)
        elif model_name == '':
            pass

        # Train model
        t1 = time.time()
        # [1:] to remove runID column, but keep it for saving results
        model_rg.fit(X_del[train_ix, :], Y[train_ix, :])
        training_time.append(time.time() -t1)

        # Predict result
        t1 = time.time()
        y_est = model_rg.predict(X_del[test_ix, :])
        predict_time.append(time.time() - t1)
        y_ests.append(y_est)

        # Calculate different error metrics
        rmse, c_rmse, mom0_err, mom0_err_rel, mom3_err, mom3_err_rel, av_len_rmse = calculate_errors(Y[test_ix,:], y_est)
        # Log errors
        errors["RMSE_tot"].append(rmse)
        errors["RMSE_c"].append(c_rmse)
        errors["mu0"].append(mom0_err)
        errors["mu0_rel"].append(mom0_err_rel)
        errors["mu3"].append(mom3_err)
        errors["mu3_rel"].append(mom3_err_rel)
        errors["av_len"].append(av_len_rmse)

        if not kFoldFlag:
            break

    errors = pd.DataFrame(errors)
    print(errors)
    return errors, training_time, predict_time, model_rg


def test_hyperparameters_rfg(max_depth_set, estimators_set, encoding, save_name):

    results = {"max_depth":[], "n_estimators":[], "training_time": [], "prediction_time": [], "RMSE_tot_mean":[], "RMSE_c_mean":[], "mu0_mean":[], "mu0_rel_mean":[], "mu3_mean":[], "mu3_rel_mean":[], "av_len_mean":[], "RMSE_tot_std":[], "RMSE_c_std":[], "mu0_std":[], "mu0_rel_std":[], "mu3_std":[], "mu3_rel_std":[], "av_len_std":[]}
    for d in max_depth_set:
        for e in estimators_set:
            hyperparameters = {'max_depth': d, 'n_estimators': e}
            print(hyperparameters)
            errors, training_time, predict_time, _ = train_test_regression(X, Y, 'RandomForestRegressor',
                                                                           hyperparameters, kFoldFlag=True, n_splits=5)
            print(errors.mean())
            # Log results
            results["max_depth"].append(d)
            results["n_estimators"].append(e)
            results["training_time"].append(np.mean(training_time))
            results["prediction_time"].append(np.mean(predict_time))
            for col in errors.columns:
                results[col+"_mean"].append(errors[col].mean())
                results[col+"_std"].append(errors[col].std())
    results = pd.DataFrame(results)
    print(results)
    if encoding:
        results.to_csv(f"data/Prediction_hyperparameter/rfg_{save_name}.csv")
    else:
        results.to_csv(f"data/Prediction_hyperparameter/rfg_{save_name}_unencoded.csv")
    return results


def regression_model_encoded(X, Y, unreliable_runID, model_name, hyperparameters, dL, test_ratio=0.1):
    """


    Returns:

    """
    # consider the unreliable runID and remove them
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

    test_middle = Y_test[:, 2]
    test_width = Y_test[:, 1]
    n_ob_test = Y_test.shape[1] - 3  # exclude c, width, middle
    # center around test_middle, generate 1/2*(n_ob_test-1) evenly spaced points, with interval test_width
    x_test_left = -test_width / 2 * (n_ob_test - 1) + test_middle * dL
    x_test_right = test_width / 2 * (n_ob_test - 1) + test_middle * dL
    x_test_loc = np.linspace(x_test_left, x_test_right, n_ob_test).T

    # get the observation locations for prediction
    pre_middle = y_pre[:, 2]
    pre_width = y_pre[:, 1]
    n_ob_pre = y_pre.shape[1] - 3  # exclude c, width, middle
    x_pre_left = -pre_width / 2 * (n_ob_pre - 1) + pre_middle * dL
    x_pre_right = pre_width / 2 * (n_ob_pre - 1) + pre_middle * dL
    x_pre_loc = np.linspace(x_pre_left, x_pre_right, n_ob_pre).T

    # Plot the result
    plt.figure(figsize=(6, 8))
    plt.suptitle(f"{hyperparameters}")
    # the distribution
    plt.subplot(3, 1, 1)
    plt.xlim((0, 500))
    plt.plot(x_test_loc[10, :], Y_test[10, 3:], label="case_one", color="b")
    plt.plot(x_pre_loc[10, :], y_pre[10, 3:], label="case_one_pred", color="g")
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

    plt.show()

    return


if __name__ == '__main__':

    # tune some parameters
    encoded = True
    dL = 0.5
    test_ratio = 0.1

    # save_name = "InputMat_231108_1637"  # small case (10) with only temperature varying, gaussian like
    # save_name = "InputMat_231110_1058"  # small case (10) with parameters regarding growth and nucleation constant
    # save_name = "InputMat_231111_2012"  # large case (300) with parameters regarding growth and nucleation constant
    # save_name = "InputMat_231109_1543"  # small case (10) with all varied
    # save_name = "InputMat_231021_0805"  # large case (200)
    # save_name = "InputMat_231110_0720"  # large case (300)
    # save_name = "InputMat_231115_1455"  # large case (300) with all varied and t_end =500
    save_name = "InputMat_231207_1605"  # large case (400) with all varied and t_end =5000
    # save_name = "InputMat_231213_1132"  # large case (1024) with all varied and t_end =5000

    ob_input = 91
    ob_output = 91
    import_file_input = f'D:/PycharmProjects/GMM/data/sparse_training_data/{save_name}_input_{ob_input}_{ob_output}_fixed_both.csv'
    import_file_output = f'D:/PycharmProjects/GMM/data/sparse_training_data/{save_name}_output_{ob_input}_{ob_output}_fixed_both.csv'
    X = pd.read_csv(import_file_input, index_col=0)
    Y = pd.read_csv(import_file_output, index_col=0)
    # convert to numpy array
    X = X.to_numpy()
    Y = Y.to_numpy()

    # delete the second column of X, which is the RunID
    X = np.delete(X, 1, axis=1)
    # input_size = 53
    # output_size = 93
    #
    # import_file_input = f'D:/PycharmProjects/GMM/data/sparse_training_data/{save_name}_input_{input_size}_{output_size}.csv'
    # import_file_output = f'D:/PycharmProjects/GMM/data/sparse_training_data/{save_name}_output_{input_size}_{output_size}.csv'
    # X = pd.read_csv(import_file_input, index_col=0)
    # Y = pd.read_csv(import_file_output, index_col=0)
    # # convert to numpy array
    # X = X.to_numpy()
    # Y = Y.to_numpy()

    # Hyperparameter optimization
    max_depths = [8, 10, 20, 25]
    estimators = [20, 40, 60, 80]
    # results = test_hyperparameters_rfg(max_depths, estimators, encoding=encoded, save_name=save_name)
    # print(results)
    #
    # # select the best hyperparameters with the minimum RMSE_tot_mean
    # depth_best = results.loc[results['RMSE_tot_mean'].idxmin(), 'max_depth']
    # no_estimator_best = results.loc[results['RMSE_tot_mean'].idxmin(), 'n_estimators']
    # print('the best depth is: ', depth_best)
    # print('the best n_estimator is: ', no_estimator_best)
    #
    # iterate through all the hyperparameters and plot
    print('plotting the results for all hyperparameters')
    for i in [60]:
        for j in [200]:
            print('max_depth is: ', i)
            print('n_estimators is: ', j)
            regression_model_encoded(X=X, Y=Y, unreliable_runID=[], model_name='RandomForestRegressor',
                                     hyperparameters={'max_depth': i, 'n_estimators': j},
                                     dL=dL, test_ratio=test_ratio)



