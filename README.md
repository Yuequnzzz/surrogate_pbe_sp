# User manual
## Introduction
This gives a brief overview of the project and how to use it.
## main script
* sparse_preprocessing.py
* nn_test_case.py
* train_NN.py

## sparse_preprocessing.py
This script is used to preprocess the data, converting the bin values to more sparse observation values.

The script takes the following arguments:
* -save_name: The input file name
* -valid_lower_bound: 0.001 by default
* -valid_upper_bound: 0.999 by default
* -input_n_ob_points: any integer, the model will optimize it
* -output_n_ob_points: any integer, the model will optimize it
* -dL: The distance between two observation points, 0.5 by default
* -error_threshold: The threshold for the error, 0.05 by default

### How to use it:
* make sure you have the data file in the directory defined in file_path_out and file_path_in
('D:/PycharmProjects/surrogatepbe/PBEsolver_outputs/' in example)
* change the file name in the main function
* define the initial input and output number of observation points in the main function
* run the script
* return the sparse data file in the folder 'data/sparse_training_data/' with the optimized number of observation points

## nn_test_case.py
This script is used to test the neural network model constructed by **torch**.
### For hyperparameter optimization:
* comment part 1 and uncomment part 2 in the main function
* change the save_name and ob_input and ob_output in the main function 
to make sure the model takes in the correct data file
* change the hyperparameters in part 2 of the main function if you like
* return the performance of the model under different hyperparameters in the folder 'data/Prediction_hyperparameter/'

### For testing the single model:
* comment part 2 and uncomment part 1 in the main function
* change the save_name and ob_input and ob_output in the main function
* define the no_layers and hidden_size in the main function
* return the performance of the model in the folder 'data/Prediction_hyperparameter/'

## Data
- 'InputMat_231207_1605' (400 simulations, 53 input ob points, 93 output ob point)
- 'InputMat_231213_1132' (1024 simulations, 53 input ob points, 91 output ob point)
