import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from datetime import datetime as dt
import psd_class
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import random


def load_data(save_name):
    """Loads training data

    Args:
        save_name (str): Name under which training data was saved

    Returns:
        Tuple: Input matrix for PBE solver, PBE solver results
    """    
    # Input 
    input_mat = pd.read_csv(f"D:/PycharmProjects/surrogatepbe/PBEsolver_InputMatrix//{save_name}.csv")
    print("Input matrix shape: ", input_mat.shape)

    # Output
    results = {}
    for runID in input_mat["runID"]:
        try:
            results[runID] = pd.read_csv(f"D:/PycharmProjects/surrogatepbe/PBEsolver_outputs/PBEsolver_{save_name}_runID{int(runID)}.csv")
        except:
            pass
    print("PBE output files found: ", len(results))
    return input_mat, results


def reformat_input_output(input_mat, results, t_sample_frac = 0.25, no_sims=5000, shuffle = False):
    """Reformat training data to be used in training

    Args:
        input_mat (pd.Dataframe): Input Matrix for PBE solver
        results (dict): All simulation results
        t_sample_frac (float, optional): The fraction of timepoints in each simulation to sample. Defaults to 0.25.
        no_sims (int, optional): Number of simulations to use. Defaults to 5000.
        shuffle (bool, optional): Wether to shuffle dataset or not. Defaults to False.

    Returns:
        Tuple: Training data as input and output array
    """
    input_columns = ['runID','T0', 'dT', 'dt', 'S0', 'sol_k0', 'sol_kT', 'growth_k0', 'growth_kS',
       'nuc_k0', 'nuc_kS', 'ini_mu0'] + [f"inipop_bin{x}" for x in range(1000)]
    output_columns = ["c"] + [f"pop_bin{x}" for x in range(1000)]

    X, Y = [], []
    for runID, res in results.items():
        # res = res.sample(frac=t_sample_frac)

        no_timepoints = res.shape[0]
        Y.append(np.array(res[output_columns]))

        relevant_inputs = np.array(input_mat.query("runID == @runID")[input_columns])
        relevant_inputs_repeated = np.vstack([relevant_inputs]*no_timepoints)

        t_vec = np.array(res["t"])[..., np.newaxis]
        x = np.hstack([t_vec, relevant_inputs_repeated])

        X.append(x)
        if len(X) > no_sims:
            break

    X = np.vstack(X)
    Y = np.vstack(Y)

    if shuffle:
        ix = np.random.permutation(X.shape[0])
        X = X[ix,:]
        Y = Y[ix,:]

    print("X, Y dimensions: ", X.shape, Y.shape)

    return X,Y


def calculate_errors(y_actual, y_predicted):
    """Evaluate prediction accuracy with multiple metrics

    Args:
        y_actual (np.array): Actual output array
        y_predicted (np.array): Estimated output array

    Returns:
        Tuple: Various error metrics
    """
    # RMSE
    rmse = mean_squared_error(y_actual, y_predicted, squared=False)

    # moments difference
    moments_actual = []
    moments_predicted = []
    for ix in range(y_actual.shape[0]):
        pop = psd_class.PSD(f_vec=y_actual[ix,1:], L_max = 500)
        pop_pred = psd_class.PSD(f_vec=y_predicted[ix,1:], L_max = 500)
        moments_actual.append(pop.moments)
        moments_predicted.append(pop_pred.moments)
    moments_actual = np.vstack(moments_actual)
    moments_predicted = np.vstack(moments_predicted)
    av_len_actual = moments_actual[:,1]/moments_actual[:,0]
    av_len_predicted = moments_predicted[:,1]/moments_predicted[:,0]

    mom0_err = mean_squared_error(moments_actual[:,0], moments_predicted[:,0], squared=False)
    mom3_err = mean_squared_error(moments_actual[:,3], moments_predicted[:,3], squared=False)
    mom0_err_rel = np.mean(np.abs(moments_actual[:,0] - moments_predicted[:,0])/moments_actual[:,0])
    mom3_err_rel = np.mean(np.abs(moments_actual[:,3] - moments_predicted[:,3])/moments_actual[:,3])

    av_len_rmse = mean_squared_error(av_len_actual, av_len_predicted, squared=False)


    # concentration difference
    c_rmse = mean_squared_error(y_actual[:,0], y_predicted[:,0], squared=False)

    return rmse, c_rmse, mom0_err, mom0_err_rel, mom3_err, mom3_err_rel, av_len_rmse


def train_test_NN(X, Y, nodes_per_layer, layers, kFoldFlag=False, n_splits=5, saveFlag=False):
    """Train and test a single neural network with specific hyperparameters

    Args:
        X (np.array): Input array
        Y (np.array): Output array
        nodes_per_layer (int): Number of nodes per layer
        layers (int): Number of layers
        kFoldFlag (bool, optional): Wether to do kFold crossvalidation. Defaults to False.
        n_splits (int, optional): Number of splits for kFold crossvalidation. Defaults to 5.
        saveFlag (bool, optional): Wether to save testing output vectors. Defaults to False.

    Returns:
        Tuple: Errors, runtimes, the trained neural network
    """
    # remove column with run ID
    X_del = np.delete(X, 1, 1)

    kf = KFold(n_splits=n_splits)

    training_time = []
    predict_time = []
    errors = {"RMSE_tot":[], "RMSE_c":[], "mu0":[], "mu0_rel":[], "mu3":[], "mu3_rel":[], "av_len":[]}
    y_ests = []
    for k, (train_ix, test_ix) in enumerate(kf.split(X)):
        mlpr = MLPRegressor(
            hidden_layer_sizes=([nodes_per_layer] * layers),
            alpha=0
            )

        # Train model
        t1 = time.time()
        # [1:] to remove runID column, but keep it for saving results
        mlpr.fit(X_del[train_ix, :], Y[train_ix, :])
        training_time.append(time.time() -t1)

        # Predict result
        t1 = time.time()
        y_est = mlpr.predict(X_del[test_ix, :])
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

        # Save results of prediction on test set
        # todo: be more general
        if saveFlag:
            n_ob_input = 41
            n_ob_output = 35
            input_columns = ['t', 'runID', 'T0', 'dT', 'dt', 'S0', 'sol_k0', 'sol_kT', 'growth_k0', 'growth_kS',
                             'nuc_k0', 'nuc_kS', 'ini_mu0', 'width', 'middle'] + [f"ob_{x}" for x in range(n_ob_input)]
            output_columns = ["c"] + [f"ob_{x}" for x in range(n_ob_output)]
            pd.DataFrame(Y[test_ix,:], columns=output_columns).to_csv(f"Predictions/y_test_actual_fold{k}.csv")
            pd.DataFrame(X[test_ix,:], columns=input_columns).to_csv(f"Predictions/X_test_actual_fold{k}.csv")
            pd.DataFrame(y_est, columns=output_columns).to_csv(f"Predictions/y_test_predicted_fold{k}.csv")

        if not kFoldFlag:
            break

    errors = pd.DataFrame(errors)
    print(errors)
    return errors, training_time, predict_time, mlpr


def test_hyperparameters(nodes, layers, encoding, save_name = f"hyperparamOpt_{dt.now().strftime('%y%m%d_%H%M')}"):
    """Test a combination of different number of nodes per layer and layers

    Args:
        nodes (list): Values of nodes per layer to be tested
        layers (list): Vaules of layers to be tested

    Returns:
        pd.DataFrame: Table with performance of each hyperparameter combination
    """
    results = {"nodes":[], "layers":[], "training_time": [], "prediction_time": [], "RMSE_tot_mean":[], "RMSE_c_mean":[], "mu0_mean":[], "mu0_rel_mean":[], "mu3_mean":[], "mu3_rel_mean":[], "av_len_mean":[], "RMSE_tot_std":[], "RMSE_c_std":[], "mu0_std":[], "mu0_rel_std":[], "mu3_std":[], "mu3_rel_std":[], "av_len_std":[]}
    for n in nodes:
        for l in layers:
            errors, training_time, predict_time, _ = train_test_NN(X, Y, nodes_per_layer=n, layers=l, kFoldFlag=True, n_splits=5)
            print(errors.mean())
            # Log results
            results["nodes"].append(n)
            results["layers"].append(l)
            results["training_time"].append(np.mean(training_time))
            results["prediction_time"].append(np.mean(predict_time))
            for col in errors.columns:
                results[col+"_mean"].append(errors[col].mean())
                results[col+"_std"].append(errors[col].std())
    results = pd.DataFrame(results)
    print(results)
    if encoding:
        results.to_csv(f"data/Prediction_hyperparameter/{save_name}.csv")
    else:
        results.to_csv(f"data/Prediction_hyperparameter/{save_name}_unencoded.csv")
    return results


def train_predict_performance_NN(X, Y, nodes_per_layer, no_layers, dL, encoded, test_ratio=0.1):
    """
    Train and test a single neural network with specific hyperparameters
    X: input array
    Y: output array
    nodes_per_layer: number of nodes per layer
    no_layers: number of layers
    dL: the interval of the bins
    encoded: binary, whether the data is encoded
    test_ratio : the fraction of test data

    Returns:

    """
    print('nodes_per_layer = ', nodes_per_layer, 'no_layers = ', no_layers)

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
    X_test = X_del[test_runID_index, :]
    X_train = X_del[train_runID_index, :]
    Y_test = Y[test_runID_index, :]
    Y_train = Y[train_runID_index, :]

    # set up the neural network
    mlpr = MLPRegressor(hidden_layer_sizes=([nodes_per_layer] * no_layers), alpha=0)
    # train the neural network
    train_start = time.time()
    mlpr.fit(X_train, Y_train)
    training_time = time.time() - train_start
    print("Training time: ", training_time)

    # predict the result
    test_start = time.time()
    y_pre = mlpr.predict(X_test)
    predict_time = time.time() - test_start
    print("Predict time: ", predict_time)

    # Calculate different error metrics
    rmse, c_rmse, mom0_err, mom0_err_rel, mom3_err, mom3_err_rel, av_len_rmse = calculate_errors(Y_test, y_pre)
    errors = {"RMSE_tot": rmse, "RMSE_c": c_rmse, "mu0": mom0_err, "mu0_rel": mom0_err_rel, "mu3": mom3_err, "mu3_rel": mom3_err_rel, "av_len": av_len_rmse}
    errors = pd.DataFrame(errors, index=[0])
    print(errors)

    if encoded:
        # get the observation locations
        # test_middle = Y_test[:, -1]
        # test_width = Y_test[:, -2]
        test_middle = Y_test[:, 2]
        test_width = Y_test[:, 1]
        n_ob_test = Y_test.shape[1] - 3  # exclude c, width, middle
        # center around test_middle, generate 1/2*(n_ob_test-1) evenly spaced points, with interval test_width
        x_test_left = -test_width / 2 * (n_ob_test - 1) + test_middle * dL
        x_test_right = test_width / 2 * (n_ob_test - 1) + test_middle * dL
        x_test_loc = np.linspace(x_test_left, x_test_right, n_ob_test).T

        # get the observation locations for prediction
        # pre_middle = y_pre[:, -1]
        # pre_width = y_pre[:, -2]
        pre_middle = y_pre[:, 2]
        pre_width = y_pre[:, 1]
        n_ob_pre = y_pre.shape[1] - 3  # exclude c, width, middle
        x_pre_left = -pre_width / 2 * (n_ob_pre - 1) + pre_middle * dL
        x_pre_right = pre_width / 2 * (n_ob_pre - 1) + pre_middle * dL
        x_pre_loc = np.linspace(x_pre_left, x_pre_right, n_ob_pre).T

        # Plot the result
        # plot_id = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        plot_id = random.sample(range(400), 10)
        for p in plot_id:
            plt.figure(figsize=(8, 6))
            plt.plot(x_test_loc[p, :], Y_test[p, 3:], label="solver", color="b")
            plt.plot(x_pre_loc[p, :], y_pre[p, 3:], label="surrogate", color="g", ls="--")
            plt.legend(prop={'size': 25})

            plt.show()

        # plt.figure(figsize=(8, 6))
        # # in the title, include the number of nodes per layer and the number of layers
        # # plt.suptitle(f"nodes_per_layer = {nodes_per_layer}, layers = {no_layers}")
        # # the distribution
        # # plt.subplot(3, 1, 1)
        # # plt.plot(x_test_loc[11, :], Y_test[11, 1:-2], label="solver", color="b")
        # # plt.plot(x_pre_loc[11, :], y_pre[11, 1:-2], label="surrogate", color="g", ls="--")
        # plt.plot(x_test_loc[20, :], Y_test[20, 3:], label="solver", color="b")
        # plt.plot(x_pre_loc[20, :], y_pre[20, 3:], label="surrogate", color="g", ls="--")
        # plt.legend(prop={'size': 25})
        # # plt.ylabel(r"PSD $f$ [m$^{-3}\mu$m$^{-1}$]")
        # # plt.xlabel(r"size $L$ [um]")

        # plt.subplot(3, 1, 2)
        # plt.xlim((0, 500))
        # plt.plot(x_test_loc[-1, :], Y_test[-1, 1:-2], label="case_two", color="r")
        # plt.plot(x_pre_loc[-1, :], y_pre[-1, 1:-2], label="case_two_pred", color="g")
        # plt.legend()
        #
        # # the concentration through the whole process
        # plt.subplot(3, 1, 3)
        # plt.scatter(X_test[0, 0], Y_test[0, 0], label='true_concentration', color='r')
        # plt.scatter(X_test[0, 0], y_pre[0, 0], label='pred_concentration', color='g')
        # plt.xlabel(r'')
        # plt.ylabel(r"Concentration")
        # plt.legend()

        # save the plot
        # plt.savefig(f"data/Prediction_hyperparameter/fig/Encoded_figure_{dt.now().strftime('%y%m%d_%H%M')}_{nodes_per_layer}_{no_layers}.png")
        # show the plot
        plt.show()

    else:
        # plot the unencoded result
        # load x
        L_max = 500  # [um]  # todo: be more general
        L_bounds = np.arange(0, L_max + dL, dL)  # [um]
        L_mid = np.mean([L_bounds[:-1], L_bounds[1:]], axis=0)  # [um]
        x = L_mid

        plt.figure(figsize=(8, 6))
        # plt.subplot(2, 1, 1)
        plt.xlim((0, 500))
        plt.plot(x, Y_test[20, 1:], label='solver', color="r")
        plt.plot(x, y_pre[20, 1:], label="surrogate", color="g", ls="--")
        # plt.ylabel(r"PSD $f$ [m$^{-3}\mu$m$^{-1}$]")
        # plt.xlabel(r"size $L$ [um]")
        plt.legend(prop={'size': 25})

        # plt.subplot(2, 1, 2)
        # plt.xlim((0, 500))
        # plt.plot(x, Y_test[-1, 1:], label="case_two", color="r")
        # plt.plot(x, y_pre[-1, 1:], label="case_two_pred", color="g", ls="--")
        # plt.legend()
        plt.show()

    return


if __name__ == "__main__":
    kFoldFlag = False

    encoded = True
    dL = 0.5
    test_ratio = 0.1
    # save_name = "InputMat_231108_1637"  # small case (10) with only temperature varying, gaussian like
    # save_name = "InputMat_231122_0934"  # large case(100) with only temperature varying, gaussian like
    # save_name = "InputMat_231110_1058"  # small case (10) with parameters regarding growth and nucleation constant
    # save_name = "InputMat_231110_1125"  # problematic dataset(!) small case (10) with only distribution varying

    # save_name = "InputMat_231109_1543"  # small case (10) with all varied

    # save_name = "InputMat_231021_0805"  # large case (200)
    # save_name = "InputMat_231110_0720"  # large case (300)
    # save_name = "InputMat_231115_1455"  # large case (300) with all varied and t_end =500
    save_name = "InputMat_231207_1605"  # large case (400) with all varied and t_end =5000, extended grid

    if encoded:
        # load the encoded data
        # import_file_input = f'D:/PycharmProjects/GMM/data/sparse_training_data/{save_name}_input_53_69.csv'
        # import_file_output = f'D:/PycharmProjects/GMM/data/sparse_training_data/{save_name}_output_53_69.csv'
        import_file_input = f'D:/PycharmProjects/GMM/data/sparse_training_data/{save_name}_input_91_91_fixed_both.csv'
        import_file_output = f'D:/PycharmProjects/GMM/data/sparse_training_data/{save_name}_output_91_91_fixed_both.csv'
        X = pd.read_csv(import_file_input, index_col=0)
        Y = pd.read_csv(import_file_output, index_col=0)
        # convert to numpy array
        X = X.to_numpy()
        Y = Y.to_numpy()

    else:
        input_mat, results = load_data(save_name)
        X, Y = reformat_input_output(input_mat, results)

    # Hyperparameter optimization
    nodes_per_layer_set = [120, 150, 180, 200]
    layers_set = [8, 10, 15, 20]
    results = test_hyperparameters(nodes_per_layer_set, layers_set, encoding=encoded)
    print(results)

    # select the best hyperparameters with the minimum RMSE_tot_mean
    nodes_per_layer_best = results.loc[results['RMSE_tot_mean'].idxmin(), 'nodes']
    layers_best = results.loc[results['RMSE_tot_mean'].idxmin(), 'layers']
    print('the best nodes_per_layer is: ', nodes_per_layer_best)
    print('the best layers is: ', layers_best)

    # iterate through all the hyperparameters and plot
    print('plotting the results for all hyperparameters')
    for j in nodes_per_layer_set:
        for k in layers_set:
            train_predict_performance_NN(X, Y, j, k, dL=dL, encoded=encoded, test_ratio=test_ratio)
