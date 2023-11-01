import numpy as np
import pandas as pd


def load_data(save_name):
    '''
    Load training data
    :param save_name: Under which name the training data was saved
    :return: tuple, input matrix for PBE solver, PBE solver results
    '''
    # Input

    input_mat = pd.read_csv(f"data/gmm_results/{save_name}.csv")
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


def reformat_input_output(input_mat, results, t_sample_frac=0.25, no_sims=5000, shuffle=False):
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
    input_columns = ['runID', 'T0', 'dT', 'dt', 'S0', 'sol_k0', 'sol_kT', 'growth_k0', 'growth_kS',
                     'nuc_k0', 'nuc_kS', 'ini_mu0'] + [f"inipop_bin{x}" for x in range(1000)]
    output_columns = ["c"] + [f"pop_bin{x}" for x in range(1000)]

    X, Y = [], []
    for runID, res in results.items():
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
        pop = psd_class.PSD(f_vec=y_actual[ix, 1:], L_max=500)
        pop_pred = psd_class.PSD(f_vec=y_predicted[ix, 1:], L_max=500)
        moments_actual.append(pop.moments)
        moments_predicted.append(pop_pred.moments)
    moments_actual = np.vstack(moments_actual)
    moments_predicted = np.vstack(moments_predicted)
    av_len_actual = moments_actual[:, 1] / moments_actual[:, 0]
    av_len_predicted = moments_predicted[:, 1] / moments_predicted[:, 0]

    mom0_err = mean_squared_error(moments_actual[:, 0], moments_predicted[:, 0], squared=False)
    mom3_err = mean_squared_error(moments_actual[:, 3], moments_predicted[:, 3], squared=False)
    mom0_err_rel = np.mean(np.abs(moments_actual[:, 0] - moments_predicted[:, 0]) / moments_actual[:, 0])
    mom3_err_rel = np.mean(np.abs(moments_actual[:, 3] - moments_predicted[:, 3]) / moments_actual[:, 3])

    av_len_rmse = mean_squared_error(av_len_actual, av_len_predicted, squared=False)

    # concentration difference
    c_rmse = mean_squared_error(y_actual[:, 0], y_predicted[:, 0], squared=False)

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
    X_del = np.delete(X, 0, 1)

    kf = KFold(n_splits=n_splits)

    training_time = []
    predict_time = []
    errors = {"RMSE_tot": [], "RMSE_c": [], "mu0": [], "mu0_rel": [], "mu3": [], "mu3_rel": [], "av_len": []}
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
        training_time.append(time.time() - t1)

        # Predict result
        t1 = time.time()
        y_est = mlpr.predict(X_del[test_ix, :])
        predict_time.append(time.time() - t1)
        y_ests.append(y_est)

        # Calculate different error metrics
        rmse, c_rmse, mom0_err, mom0_err_rel, mom3_err, mom3_err_rel, av_len_rmse = calculate_errors(Y[test_ix, :],
                                                                                                     y_est)
        # Log errors
        errors["RMSE_tot"].append(rmse)
        errors["RMSE_c"].append(c_rmse)
        errors["mu0"].append(mom0_err)
        errors["mu0_rel"].append(mom0_err_rel)
        errors["mu3"].append(mom3_err)
        errors["mu3_rel"].append(mom3_err_rel)
        errors["av_len"].append(av_len_rmse)

        # Save results of prediction on test set
        if saveFlag:
            input_columns = ['t', 'runID', 'T0', 'dT', 'dt', 'S0', 'sol_k0', 'sol_kT', 'growth_k0', 'growth_kS',
                             'nuc_k0', 'nuc_kS', 'ini_mu0'] + [f"inipop_bin{x}" for x in range(1000)]
            output_columns = ["c"] + [f"pop_bin{x}" for x in range(1000)]
            pd.DataFrame(Y[test_ix, :], columns=output_columns).to_csv(f"Predictions/y_test_actual_fold{k}.csv")
            pd.DataFrame(X[test_ix, :], columns=input_columns).to_csv(f"Predictions/X_test_actual_fold{k}.csv")
            pd.DataFrame(y_est, columns=output_columns).to_csv(f"Predictions/y_test_predicted_fold{k}.csv")

        if not kFoldFlag:
            break

    errors = pd.DataFrame(errors)
    print(errors)
    return errors, training_time, predict_time, mlpr


def test_hyperparameters(nodes, layers, save_name=f"hyperparamOpt_{dt.now().strftime('%y%m%d_%H%M')}"):
    """Test a combination of different number of nodes per layer and layers

    Args:
        nodes (list): Values of nodes per layer to be tested
        layers (list): Vaules of layers to be tested

    Returns:
        pd.DataFrame: Table with performance of each hyperparameter combination
    """
    results = {"nodes": [], "layers": [], "training_time": [], "prediction_time": [], "RMSE_tot_mean": [],
               "RMSE_c_mean": [], "mu0_mean": [], "mu0_rel_mean": [], "mu3_mean": [], "mu3_rel_mean": [],
               "av_len_mean": [], "RMSE_tot_std": [], "RMSE_c_std": [], "mu0_std": [], "mu0_rel_std": [], "mu3_std": [],
               "mu3_rel_std": [], "av_len_std": []}
    for n in nodes:
        for l in layers:
            errors, training_time, predict_time, _ = train_test_NN(X, Y, nodes_per_layer=n, layers=l, kFoldFlag=True,
                                                                   n_splits=5)
            print(errors.mean())
            # Log results
            results["nodes"].append(n)
            results["layers"].append(l)
            results["training_time"].append(np.mean(training_time))
            results["prediction_time"].append(np.mean(predict_time))
            for col in errors.columns:
                results[col + "_mean"].append(errors[col].mean())
                results[col + "_std"].append(errors[col].std())
    results = pd.DataFrame(results)
    print(results)
    results.to_csv(f"Predictions/{save_name}.csv")
    return results


if __name__ == "__main__":
    kFoldFlag = False

    save_name = "InputMat_231004_1018"
    input_mat, results = load_data(save_name)

    X, Y = reformat_input_output(input_mat, results)

    # Train single model
    errors, training_time, predict_time, mlpr = train_test_NN(X, Y, nodes_per_layer=100, layers=10, kFoldFlag=True,
                                                              n_splits=5, saveFlag=True)
    print("Training time: ", np.mean(training_time))

    # Hyperparameter optimization
    results = test_hyperparameters([10, 20], [2])
    print(results)