import random
import numpy as np
import pandas as pd
import time

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import TensorDataset, DataLoader

from base_functions import *
from datetime import datetime as dt


############## TORCH NN ##############


def normalize(X, type):
    if type == 'relu':
        return (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    elif type == 'sigmoid':
        return (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0)) * 2 - 1


def un_normalize(X_norm, min, max, type):
    if type == 'relu':
        return X_norm * (max - min) + min
    elif type == 'sigmoid':
        return (X_norm + 1)/2 * (max - min) + min


# Define a simple neural network model
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, no_layers, output_size, type):
        super(NeuralNetwork, self).__init__()
        
        layers = []
        activation = None
        if type == 'relu':
            activation = nn.ReLU()
        elif type == 'sigmoid':
            activation = nn.Sigmoid()
        # First Layer
        layers += [nn.Linear(input_size, hidden_size), activation]
        # Hidden Layers
        layers += [nn.Linear(hidden_size, hidden_size), activation]*no_layers
        # Output Layer 
        layers += [nn.Linear(hidden_size, output_size), activation]
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        return x


def train(model, dataloader, optimizer, criterion, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    
    return model


def compute_moments(y, dL=0.5):
    # unify the data format
    y_array = y.numpy()

    middle = y_array[:, -1]
    width = y_array[:, -2]
    n_ob = y_array.shape[1] - 3  # exclude c, width, middle
    L_left = -width / 2 * (n_ob - 1) + middle * dL
    L_right = width / 2 * (n_ob - 1) + middle * dL
    Lengths = np.linspace(L_left, L_right, n_ob).T

    mu0 = np.sum(Lengths ** 0 * y_array[:, 1:-2] * width[:, None], axis=1)
    mu1 = np.sum(Lengths ** 1 * y_array[:, 1:-2] * width[:, None], axis=1)
    mu2 = np.sum(Lengths ** 2 * y_array[:, 1:-2] * width[:, None], axis=1)
    mu3 = np.sum(Lengths ** 3 * y_array[:, 1:-2] * width[:, None], axis=1)
    return mu0, mu1, mu2, mu3


def my_loss(output, target):
    # Mean squared error based, but more weight on the middle and width
    loss = torch.mean((output - target)**2)
    # m3_ini = compute_moments(target)[3]
    # m3 = compute_moments(output)[3]
    # c_total_ini =
    # loss = 10 * torch.mean((output[:, -1] - target[:, -1])**2) + 10 * torch.mean((output[:, -2] - target[:, -2])**2) + \
    #        torch.mean((output[:, 1:-2] - target[:, 1:-2])**2) + 10 * torch.mean((output[:, 0] - target[:, 0])**2)
    return loss


def construct_nn_model(X_train, y_train, X_test, input_size, hidden_size, no_layers, output_size, num_epochs, batch_size, choose_type, learning_rate=0.001):
    """
    construct the neural network model
    :return:
    """
    model = NeuralNetwork(input_size, hidden_size, no_layers, output_size, choose_type)
    print('model is defined')

    # Define the loss function and optimizer
    criterion = my_loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print('start training')
    t1 = time.time()
    model = train(model, dataloader, optimizer, criterion, num_epochs)
    t2 = time.time()
    training_time = t2 - t1
    print('training time is ', training_time)

    # Test the model
    print('start testing')
    with torch.no_grad():
        model.eval()
        X_test = torch.tensor(X_test, dtype=torch.float32)  # Convert test data to PyTorch tensor
        y_est = model(X_test)

    t3 = time.time()
    predict_time = t3 - t2
    return y_est, training_time, predict_time


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
    # concentration difference
    c_rmse = mean_squared_error(y_actual[:, 0], y_predicted[:, 0], squared=False)
    # width difference
    width_rmse = mean_squared_error(y_actual[:, -2], y_predicted[:, -2], squared=False)
    # middle difference
    middle_rmse = mean_squared_error(y_actual[:, -1], y_predicted[:, -1], squared=False)

    # moments difference
    mu0, mu1, mu2, mu3 = compute_moments(y_actual)
    mu0_pred, mu1_pred, mu2_pred, mu3_pred = compute_moments(y_predicted)
    mom0_rmse = mean_squared_error(mu0, mu0_pred, squared=False)
    mom1_rmse = mean_squared_error(mu1, mu1_pred, squared=False)
    mom3_rmse = mean_squared_error(mu3, mu3_pred, squared=False)
    mom0_rel = np.mean(np.abs(mu0 - mu0_pred)/mu0)
    mom1_rel = np.mean(np.abs(mu1 - mu1_pred)/mu1)
    mom3_rel = np.mean(np.abs(mu3 - mu3_pred)/mu3)

    return rmse, c_rmse, width_rmse, middle_rmse, mom0_rmse, mom0_rel, mom1_rmse, mom1_rel, mom3_rmse, mom3_rel


def cross_validation(X, Y, no_layers, nodes, n_splits=5):
    # Delete the second column of X, which is the RunID
    X_del = np.delete(X, 1, 1)
    # ----------------------------------------------
    # Get min and max values for un-normalizing
    min_y = Y.min(axis=0)
    max_y = Y.max(axis=0)
    # Normalize the data todo: see what happens
    X_del = normalize(X_del, 'relu')
    Y = normalize(Y, 'relu')
    # ----------------------------------------------

    # Define the K-fold Cross Validator
    kf = KFold(n_splits=n_splits)
    # kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    training_time = []
    predict_time = []
    errors = {
        "RMSE_tot": [],
        "RMSE_c": [],
        "RMSE_width": [],
        "RMSE_middle": [],
        "mu0": [],
        "mu0_rel": [],
        "mu3": [],
        "mu3_rel": [],
        "av_len": [],
        "av_len_rel": []}
    y_ests = []

    for k, (train_ix, test_ix) in enumerate(kf.split(X)):
        X_train, X_test = X_del[train_ix, :], X_del[test_ix, :]
        y_train, y_test = Y[train_ix, :], Y[test_ix, :]
        # # todo: shuffle the training data
        # X_train, y_train = shuffle(X_train, y_train, random_state=42)
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)
        # Define the model
        input_size = X_del.shape[1]
        output_size = Y.shape[1]
        hidden_size = nodes
        no_layers = no_layers
        y_est, t_train, t_predict = construct_nn_model(
            X_train,
            y_train,
            X_test,
            input_size,
            hidden_size,
            no_layers,
            output_size,
            num_epochs=100,
            batch_size=32,
            choose_type='relu',
            learning_rate=0.001
        )

        # ----------------------------------------------
        # Un-normalize
        y_test = un_normalize(y_test, min_y, max_y, 'relu')
        y_est = un_normalize(y_est, min_y, max_y, 'relu')
        # ----------------------------------------------

        training_time.append(t_train)
        predict_time.append(t_predict)
        y_ests.append(y_est)

        # Calculate different error metrics
        rmse, c_rmse, width_rmse, middle_rmse, mom0_rmse, mom0_rel, mom1_rmse, mom1_rel, mom3_rmse, mom3_rel = calculate_errors(y_test,
                                                                                                       y_est)
        # Log errors
        errors["RMSE_tot"].append(rmse)
        errors["RMSE_c"].append(c_rmse)
        errors["RMSE_width"].append(width_rmse)
        errors["RMSE_middle"].append(middle_rmse)
        errors["mu0"].append(mom0_rmse)
        errors["mu0_rel"].append(mom0_rel)
        errors["mu3"].append(mom3_rmse)
        errors["mu3_rel"].append(mom3_rel)
        errors["av_len"].append(mom1_rmse)
        errors["av_len_rel"].append(mom1_rel)

    errors = pd.DataFrame(errors)
    print(errors)
    return errors, training_time, predict_time


def test_hyperparameters(nodes, layers, save_name = f"hyperparamOpt_{dt.now().strftime('%y%m%d_%H%M')}"):
    """Test a combination of different number of nodes per layer and layers

    Args:
        nodes (list): Values of nodes per layer to be tested
        layers (list): Vaules of layers to be tested

    Returns:
        pd.DataFrame: Table with performance of each hyperparameter combination
    """
    results = {"nodes": [], "layers": [], "training_time": [], "prediction_time": [], "RMSE_tot_mean": [],
               "RMSE_c_mean": [], "RMSE_width_mean": [], "RMSE_middle_mean": [], "mu0_mean": [], "mu0_rel_mean": [],
               "av_len_mean": [], "av_len_rel_mean": [], "mu3_mean": [], "mu3_rel_mean":[], "RMSE_tot_std": [],
               "RMSE_c_std": [], "RMSE_width_std": [], "RMSE_middle_std": [], "mu0_std": [], "mu0_rel_std": [],
               "av_len_std": [], "av_len_rel_std": [], "mu3_std": [], "mu3_rel_std": []}
    for n in nodes:
        for l in layers:
            print(f"Testing {n} nodes and {l} layers")
            errors, training_time, predict_time = cross_validation(X, Y, l, n)
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
    # Save results
    results.to_csv(f"data/Prediction_hyperparameter/{save_name}.csv")

    return results


if __name__ == "__main__":

    # -----------------load the encoded data-----------------------
    # save_name = 'InputMat_231207_1605'
    save_name = 'InputMat_231213_1132'
    ob_input = 53
    ob_output = 91
    import_file_input = f'D:/PycharmProjects/GMM/data/sparse_training_data/{save_name}_input_{ob_input}_{ob_output}.csv'
    import_file_output = f'D:/PycharmProjects/GMM/data/sparse_training_data/{save_name}_output_{ob_input}_{ob_output}.csv'
    X = pd.read_csv(import_file_input, index_col=0)
    Y = pd.read_csv(import_file_output, index_col=0)
    # convert to numpy array
    X = X.to_numpy()
    Y = Y.to_numpy()

    # # ----------------part 1: test single model----------------
    # # delete the second column of X, which is the RunID
    # X = np.delete(X, 1, axis=1)
    #
    # print('data is loaded', X.shape, Y.shape)
    #
    # # choose the type
    # choose_type = 'relu'
    #
    # # normalize the data
    # X_norm = normalize(X, choose_type)
    # Y_norm = normalize(Y, choose_type)
    #
    # # Get min and max values for un-normalizing
    # min_y = Y.min(axis=0)
    # max_y = Y.max(axis=0)
    #
    # # Convert data to PyTorch tensors
    # X_norm = torch.tensor(X_norm, dtype=torch.float32)
    # Y_norm = torch.tensor(Y_norm, dtype=torch.float32)
    #
    # # Split the data into training and testing
    # X_train_norm, X_test_norm, y_train_norm, y_test_norm = train_test_split(X_norm, Y_norm, test_size=0.2, random_state=42)
    #
    # # Define the model
    # input_size = X.shape[1]
    # output_size = Y.shape[1]
    # hidden_size = 80
    # no_layers = 4
    # model = NeuralNetwork(input_size, hidden_size, no_layers, output_size, choose_type)
    # print('model is defined')
    #
    # # Define the loss function and optimizer
    # criterion = my_loss
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    #
    # # Train the model
    # num_epochs = 100
    # batch_size = 32
    #
    # train_dataset = TensorDataset(X_train_norm, y_train_norm)
    # dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # print('start training')
    # t1 = time.time()
    # model = train(model, dataloader, optimizer, criterion, num_epochs)
    # t2 = time.time()
    # print('training time is ', t2 - t1)
    #
    # # Test the model
    # print('start testing')
    # with torch.no_grad():
    #     model.eval()
    #     X_test_norm = torch.tensor(X_test_norm, dtype=torch.float32)  # Convert test data to PyTorch tensor
    #     y_est_norm = model(X_test_norm)
    #
    # # Un-normalize
    # y_test = un_normalize(y_test_norm, min_y, max_y, choose_type)
    # y_est = un_normalize(y_est_norm, min_y, max_y, choose_type)
    # #
    #
    # #----------------------------------------------
    # # # todo: what if we use the un-normalized data
    # # X = torch.tensor(X, dtype=torch.float32)
    # # Y = torch.tensor(Y, dtype=torch.float32)
    # # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    # # input_size = X.shape[1]
    # # output_size = Y.shape[1]
    # # hidden_size = 100
    # # no_layers = 8
    # # model = NeuralNetwork(input_size, hidden_size, no_layers, output_size, choose_type)
    # # print('model is defined')
    # #
    # # # Define the loss function and optimizer
    # # criterion = my_loss
    # # optimizer = optim.Adam(model.parameters(), lr=0.001)
    # #
    # # # Train the model
    # # num_epochs = 1000
    # # batch_size = 32
    # #
    # # train_dataset = TensorDataset(X_train, y_train)
    # # dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # # print('start training')
    # # t1 = time.time()
    # # model = train(model, dataloader, optimizer, criterion, num_epochs)
    # # t2 = time.time()
    # # print('training time is ', t2 - t1)
    # #
    # # # Test the model
    # # print('start testing')
    # # with torch.no_grad():
    # #     model.eval()
    # #     X_test = torch.tensor(X_test, dtype=torch.float32)  # Convert test data to PyTorch tensor
    # #     y_est = model(X_test)
    # #
    # # # ----------------------------------------------
    # print("Output Range Torch: ")
    # print(y_est.min(), y_est.max())
    # print("Error: ")
    # print('RMSE =', error_functions(y_test, y_est))
    # pd.DataFrame(y_est).to_csv(f'data/prediction_results/fixed_est_{save_name}_both_largernn.csv')
    # pd.DataFrame(y_test).to_csv(f'data/prediction_results/fixed_test_{save_name}_both_largernn.csv')
    #
    # y_est = pd.read_csv(f'data/prediction_results/fixed_est_{save_name}_both_largernn.csv', index_col=0).to_numpy()
    # y_test = pd.read_csv(f'data/prediction_results/fixed_test_{save_name}_both_largernn.csv', index_col=0).to_numpy()
    #
    # # Plot
    # test_middle = y_test[:, -1]
    # test_width = y_test[:, -2]
    # n_ob_test = y_test.shape[1] - 3  # exclude c, width, middle
    # dL = 0.5
    #
    # # center around test_middle, generate 1/2*(n_ob_test-1) evenly spaced points, with interval test_width
    # x_test_left = -test_width / 2 * (n_ob_test - 1) + test_middle * dL
    # x_test_right = test_width / 2 * (n_ob_test - 1) + test_middle * dL
    # x_test_loc = np.linspace(x_test_left, x_test_right, n_ob_test).T
    #
    # # get the observation locations for prediction
    # pre_middle = y_est[:, -1]
    # pre_width = y_est[:, -2]
    # n_ob_pre = y_est.shape[1] - 3  # exclude c, width, middle
    # x_pre_left = -pre_width / 2 * (n_ob_pre - 1) + pre_middle * dL
    # x_pre_right = pre_width / 2 * (n_ob_pre - 1) + pre_middle * dL
    # x_pre_loc = np.linspace(x_pre_left, x_pre_right, n_ob_pre).T
    #
    # # Plot the result
    # # plot_id = random.sample(range(400), 10)
    # plot_id = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # for p in plot_id:
    #
    #     plt.figure(figsize=(8, 6))
    #     plt.title("1000")
    #     plt.plot(x_test_loc[p, :], y_test[p, 1: -2], label="solver", color="b")
    #     plt.plot(x_pre_loc[p, :], y_est[p, 1: -2], label="surrogate", color="g", ls="--")
    #     # plot a vertical line to indicate the middle
    #     plt.axvline(x=test_middle[p] * dL, color='b')
    #     plt.axvline(x=pre_middle[p] * dL, color='g')
    #
    #     plt.legend(prop={'size': 25})
    #
    #     plt.show()

    # ----------------part 2: optimize hyperparameters----------------
    nodes = [20, 50, 80]
    layers = [4, 6, 8]
    test_hyperparameters(nodes, layers)






