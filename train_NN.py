import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from datetime import datetime as dt
import psd_class
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


def load_data(save_name):
    """Loads training data

    Args:
        save_name (str): Name under which training data was saved

    Returns:
        Tuple: Input matrix for PBE solver, PBE solver results
    """    
    # Input 
    input_mat = pd.read_csv(f"PBEsolver_InputMatrix/{save_name}.csv")
    print("Input matrix shape: ", input_mat.shape)

    # Output
    results = {}
    for runID in input_mat["runID"]:
        try:
            results[runID] = pd.read_csv(f"PBEsolver_outputs/PBEsolver_{save_name}_runID{int(runID)}.csv")
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
        res = res.sample(frac=t_sample_frac)

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
        print(k)
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


def test_hyperparameters(nodes, layers, save_name = f"hyperparamOpt_{dt.now().strftime('%y%m%d_%H%M')}"):
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
    results.to_csv(f"Predictions/{save_name}.csv")
    return results


def train_predict_performance_NN(X, Y, nodes_per_layer, no_layers, dL, test_ratio=0.1):
    """
    Train and test a single neural network with specific hyperparameters
    X: input array
    Y: output array
    nodes_per_layer: number of nodes per layer
    no_layers: number of layers
    test_ratio : the fraction of test data

    Returns:

    """
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

    # get the observation locations
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
    # the distribution
    plt.figure(figsize=(6, 8))
    plt.subplot(3, 1, 1)
    plt.xlim((0, 500))
    plt.plot(x_test_loc[0, :], Y_test[0, 1:-2], label="case_one", color="b")
    plt.plot(x_pre_loc[0, :], y_pre[0, 1:-2], label="case_one_pred", color="g")
    plt.legend()
    plt.ylabel(r"PSD $f$ [m$^{-3}\mu$m$^{-1}$]")

    plt.subplot(3, 1, 2)
    plt.xlim((0, 500))
    plt.plot(x_test_loc[-1, :], Y_test[-1, 1:-2], label="case_two", color="r")
    plt.plot(x_pre_loc[-1, :], y_pre[-1, 1:-2], label="case_two_pred", color="g", ls="--")
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


if __name__ == "__main__":
    kFoldFlag = False

    # # save_name = "InputMat_231018_1624"  # small case
    # save_name = "InputMat_231021_0805"  # large case
    # input_mat, results = load_data(save_name)
    #
    # X,Y = reformat_input_output(input_mat, results)

    import_file_input = 'D:/PycharmProjects/GMM/data/sparse_training_data/InputMat_231021_0805_input_41_35.csv'
    import_file_output = 'D:/PycharmProjects/GMM/data/sparse_training_data/InputMat_231021_0805_output_41_35.csv'
    X = pd.read_csv(import_file_input, index_col=0)
    Y = pd.read_csv(import_file_output, index_col=0)


    # convert to numpy array
    X = X.to_numpy()
    Y = Y.to_numpy()

    # # Train single model
    # print('start training')
    # errors, training_time, predict_time, mlpr = train_test_NN(X, Y, nodes_per_layer = 100, layers = 10, kFoldFlag=True, n_splits=5, saveFlag=True)
    # print("Training time: ", np.mean(training_time))
    #
    # # Hyperparameter optimization
    # results = test_hyperparameters([5, 10, 20, 50, 100], [2, 4, 6, 8, 10])
    # print(results)

    # Train and test model with specific hyperparameters
    nodes_per_layer = 10
    layers = 10
    train_predict_performance_NN(X, Y, nodes_per_layer, layers, dL=0.5, test_ratio=0.1)