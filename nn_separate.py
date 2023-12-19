import os
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

    # todo: just use part of the data as output
    Y_dist = Y[:, 1:-2]
    Y_c = Y[:, 0].reshape(-1, 1)  # modify the output size to 1
    Y_loc = Y[:, -2:]
    output_dict = {'distribution': Y_dist, 'concentration': Y_c, 'location': Y_loc}

    for key, value in output_dict.items():
        print('start training', key)
        y_est, y_test = main(X, value, choose_type, hidden_size=50, no_layers=8, num_epochs=100,
                             batch_size=32, lr=0.001, normalize=False)
        # save the results
        pd.DataFrame(y_est).to_csv(f'data/prediction_results/{key}_est.csv')
        pd.DataFrame(y_test).to_csv(f'data/prediction_results/{key}_test.csv')
