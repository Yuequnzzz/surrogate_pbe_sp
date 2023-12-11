import random
import numpy as np
import pandas as pd
import time

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

from base_functions import *


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


def my_loss(output, target):
    # Mean squared error based, but more weight on the middle and width
    loss = torch.mean((output - target)**2)
    # loss = 0.1 * torch.mean((output[:, -1] - target[:, -1])**2) + 0.1 * torch.mean((output[:, -2] - target[:, -2])**2) + \
    #        torch.mean((output[:, :-2] - target[:, :-2])**2)
    return loss


if __name__ == "__main__":

    # load the encoded data
    save_name = 'InputMat_231207_1605'
    ob_input = 53
    ob_output = 93
    import_file_input = f'D:/PycharmProjects/GMM/data/sparse_training_data/{save_name}_input_{ob_input}_{ob_output}.csv'
    import_file_output = f'D:/PycharmProjects/GMM/data/sparse_training_data/{save_name}_output_{ob_input}_{ob_output}.csv'
    X = pd.read_csv(import_file_input, index_col=0)
    Y = pd.read_csv(import_file_output, index_col=0)
    # convert to numpy array
    X = X.to_numpy()
    Y = Y.to_numpy()
    print('data is loaded', X.shape, Y.shape)

    # choose the type
    choose_type = 'relu'

    # normalize the data
    X_norm = normalize(X, choose_type)
    Y_norm = normalize(Y, choose_type)

    # Get min and max values for un-normalizing
    min_y = Y.min(axis=0)
    max_y = Y.max(axis=0)

    # Convert data to PyTorch tensors
    X_norm = torch.tensor(X_norm, dtype=torch.float32)
    Y_norm = torch.tensor(Y_norm, dtype=torch.float32)

    # Split the data into training and testing
    X_train_norm, X_test_norm, y_train_norm, y_test_norm = train_test_split(X_norm, Y_norm, test_size=0.2, random_state=42)

    # Define the model
    input_size = X.shape[1]
    output_size = Y.shape[1]
    hidden_size = 50
    no_layers = 8
    model = NeuralNetwork(input_size, hidden_size, no_layers, output_size, choose_type)
    print('model is defined')

    # Define the loss function and optimizer
    criterion = my_loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    num_epochs = 100
    batch_size = 32

    train_dataset = TensorDataset(X_train_norm, y_train_norm)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print('start training')
    t1 = time.time()
    model = train(model, dataloader, optimizer, criterion, num_epochs)
    t2 = time.time()
    print('training time is ', t2 - t1)

    # Test the model
    print('start testing')
    with torch.no_grad():
        model.eval()
        X_test_norm = torch.tensor(X_test_norm, dtype=torch.float32)  # Convert test data to PyTorch tensor
        y_est_norm = model(X_test_norm)

    # Un-normalize
    y_test = un_normalize(y_test_norm, min_y, max_y, choose_type)
    y_est = un_normalize(y_est_norm, min_y, max_y, choose_type)

    print("Output Range Torch: ")
    print(y_est.min(), y_est.max())
    print("Error: ")
    print('RMSE =', error_functions(y_test, y_est))

    # Plot
    test_middle = y_test[:, -1]
    test_width = y_test[:, -2]
    n_ob_test = y_test.shape[1] - 3  # exclude c, width, middle
    dL = 0.5

    # center around test_middle, generate 1/2*(n_ob_test-1) evenly spaced points, with interval test_width
    x_test_left = -test_width / 2 * (n_ob_test - 1) + test_middle * dL
    x_test_right = test_width / 2 * (n_ob_test - 1) + test_middle * dL
    x_test_loc = np.linspace(x_test_left, x_test_right, n_ob_test).T

    # get the observation locations for prediction
    pre_middle = y_est[:, -1]
    pre_width = y_est[:, -2]
    n_ob_pre = y_est.shape[1] - 3  # exclude c, width, middle
    x_pre_left = -pre_width / 2 * (n_ob_pre - 1) + pre_middle * dL
    x_pre_right = pre_width / 2 * (n_ob_pre - 1) + pre_middle * dL
    x_pre_loc = np.linspace(x_pre_left, x_pre_right, n_ob_pre).T

    # Plot the result
    # plot_id = random.sample(range(400), 10)
    plot_id = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    for p in plot_id:

        plt.figure(figsize=(8, 6))
        plt.plot(x_test_loc[p, :], y_test[p, 1:-2], label="solver", color="b")
        plt.plot(x_pre_loc[p, :], y_est[p, 1:-2], label="surrogate", color="g", ls="--")
        plt.legend(prop={'size': 25})

        plt.show()





