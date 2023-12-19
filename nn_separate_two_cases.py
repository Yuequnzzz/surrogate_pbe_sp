import os
import pickle
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'nn_test_case.py')))

from nn_test_case import *


def main(X, Y, type, hidden_size, no_layers, num_epochs, batch_size, lr, normalize):
    # choose the type
    choose_type = type

    if normalize:
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
        X_train_norm, X_test_norm, y_train_norm, y_test_norm = train_test_split(X_norm, Y_norm, test_size=0.2,
                                                                                random_state=42)

        # Define the model
        input_size = X.shape[1]
        output_size = Y.shape[1]
        model = NeuralNetwork(input_size, hidden_size, no_layers, output_size, choose_type)
        print('model is defined')

        # Define the loss function and optimizer
        criterion = my_loss
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Train the model
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

    else:
        X = torch.tensor(X, dtype=torch.float32)
        Y = torch.tensor(Y, dtype=torch.float32)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
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

        train_dataset = TensorDataset(X_train, y_train)
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
            X_test = torch.tensor(X_test, dtype=torch.float32)  # Convert test data to PyTorch tensor
            y_est = model(X_test)

    print("Output Range Torch: ")
    print(y_est.min(), y_est.max())
    print("Error: ")
    print('RMSE =', error_functions(y_test, y_est))
    return y_est, y_test


if __name__ == '__main__':
    # load the encoded data
    save_name = 'InputMat_231207_1605'
    ob_points = 91
    import_file_input = f'D:/PycharmProjects/GMM/data/sparse_training_data/{save_name}_input_{ob_points}_{ob_points}.csv'
    import_file_output = f'D:/PycharmProjects/GMM/data/sparse_training_data/{save_name}_output_{ob_points}_{ob_points}.csv'
    X = pd.read_csv(import_file_input, index_col=0)
    Y = pd.read_csv(import_file_output, index_col=0)

    # check if the data contains nan
    print('check if the data contains nan')
    print(X.isnull().values.any())
    print(Y.isnull().values.any())

    # find where the nan is
    print('find where the nan is')
    nan_df = X.isna()
    nan_df = nan_df.any(axis=1)
    print(nan_df)


    # # load the both_ids, to separate the data if needed
    # with open(f'data/sparse_training_data/both_ids_{save_name}.pkl', 'rb') as f:
    #     both_ids = pickle.load(f)
    # # convert to numpy array
    # X = X.to_numpy()
    # Y = Y.to_numpy()
    # print('data is loaded', X.shape, Y.shape)
    #
    # # choose the type
    # choose_type = 'relu'
    #
    # # reorganize the data and structure the model
    # X = torch.tensor(X, dtype=torch.float32)
    # Y = torch.tensor(Y, dtype=torch.float32)
    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    # input_size = X.shape[1]
    # output_size = Y.shape[1]
    # hidden_size = 50
    # no_layers = 8
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
    # train_dataset = TensorDataset(X_train, y_train)
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
    #     X_test = torch.tensor(X_test, dtype=torch.float32)  # Convert test data to PyTorch tensor
    #     y_est = model(X_test)
    #
    # # ----------------------------------------------
    # print("Output Range Torch: ")
    # print(y_est.min(), y_est.max())
    # print("Error: ")
    # print('RMSE =', error_functions(y_test, y_est))
    #
    # # ----------------------------------------------
    # pd.DataFrame(y_est).to_csv(f'data/prediction_results/fixed_est_{save_name}.csv')
    # pd.DataFrame(y_test).to_csv(f'data/prediction_results/fixed_test_{save_name}.csv')
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
    #     plt.plot(x_test_loc[p, :], y_test[p, 1:-2], label="solver", color="b")
    #     plt.plot(x_pre_loc[p, :], y_est[p, 1:-2], label="surrogate", color="g", ls="--")
    #     plt.legend(prop={'size': 25})
    #
    #     plt.show()