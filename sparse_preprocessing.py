import numpy as np
import pandas as pd
import scipy as sp
import copy
from base_functions import *
import matplotlib.pyplot as plt


def search_for_valid_region(cdf_data, lower_bound, upper_bound):
    """
    search for the valid region
    :param cdf_data: the cdf data
    :param lower_bound: the lower bound of the valid region
    :param upper_bound: the upper bound of the valid region
    :return: the valid region
    """
    # initialize
    middle_id = np.zeros(cdf_data.shape[0])
    start_id = np.zeros(cdf_data.shape[0])
    end_id = np.zeros(cdf_data.shape[0])
    # search for the bounds in cdf
    for i in range(cdf_data.shape[0]):
        pos = np.where((cdf_data.iloc[i, :] > lower_bound) & (cdf_data.iloc[i, :] < upper_bound))[0]
        # get the middle point of the valid region
        middle_id[i] = np.median(pos)
        start_id[i] = pos[0]
        end_id[i] = pos[-1]
    # convert the type to int
    middle_id = middle_id.astype(int)
    start_id = start_id.astype(int)
    end_id = end_id.astype(int)
    return start_id, middle_id, end_id


def place_ob_points(start_id, end_id, n_ob_points, dL):
    """
    place the observation points
    :param start_id: the start id of the valid region
    :param end_id: the end id of the valid region
    :param n_ob_points: the number of observation points
    :param dL: the width of each bin
    :return: the id of observation points, middle points, and the width between the observation points
    """
    # make sure the number of observation points is odd
    if n_ob_points % 2 == 0:
        raise ValueError('The number of observation points should be odd')
    # initialize
    ob_points_id = np.zeros((start_id.shape[0], n_ob_points))
    middle_id = np.zeros(start_id.shape[0])
    # place the observation points
    for i in range(start_id.shape[0]):
        # find the data points in symmetric position
        ob_points_id[i, :] = np.linspace(start_id[i], end_id[i], n_ob_points)
        # get the middle point
        middle_id[i] = np.median(ob_points_id[i, :])
    # get the width
    width_in_x = (ob_points_id[:, 1] - ob_points_id[:, 0]) * dL
    return ob_points_id, middle_id, width_in_x


def real_dist_ob(psd_data, ob_points_id):
    """
    get the real probability of the observation points
    :param psd_data: the psd data
    :param ob_points_id: the id of observation points
    :return: the real probability of the observation points
    """
    # initialize
    ob_psd = np.zeros((psd_data.shape[0], ob_points_id.shape[1]))
    # get the real probability by interpolation
    for i in range(psd_data.shape[0]):
        ob_psd[i, :] = np.interp(ob_points_id[i, :], np.arange(psd_data.shape[1]), psd_data.iloc[i, :].tolist())
    return ob_psd


def predict_psd_extrapolate(psd_data, ob_points_id, ob_psd):
    """
    Estimate how good the fitting performance is by calculating the error
    :param psd_data: the psd data
    :param ob_points_id: the id of observation points
    :param ob_psd: the real probability of the observation points
    :return: the predicted pdf
    """
    # initialize the error
    y_pre = np.zeros(psd_data.shape)
    # use the observation points via interpolating to calculate the predicted probability at the original position
    for i in range(ob_points_id.shape[0]):
        y_pre[i, :] = sp.interpolate.interp1d(ob_points_id[i, :], ob_psd[i, :],
                                              fill_value='extrapolate')(np.arange(psd_data.shape[1]))
        # replace the negative value with 0
        y_pre[i, y_pre[i, :] < 0] = 0
    return y_pre


def error_extrapolate(data_psd, y_pre):
    """
    Estimate how good the fitting performance is by calculating the error
    :param data_psd: the pdf data
    :param y_pre: the predicted pdf
    :return: the error
    """
    # initialize the error
    error = np.zeros(data_psd.shape[0])
    # calculate the error
    for i in range(data_psd.shape[0]):
        # error[i] = error_func_relative(np.asarray(data_pdf.iloc[i, :]), y_pre[i, :].reshape(-1, 1))
        # error[i] = error_func_smape(data_psd.iloc[i, :], y_pre[i, :].reshape(-1, 1))
        error[i] = error_func_scaled_ae(data_psd.iloc[i, :], y_pre[i, :].reshape(-1, 1))
    print(f"the max error  is {np.max(error)}, occur at line {np.argmax(error)}")
    print(f"the min error  is {np.min(error)}, occur at line {np.argmin(error)}")
    print(f"the error of the last line is {error[-1]}")
    return error


def sparse_model(data_bins, valid_lower_bound, valid_upper_bound, n_ob_points, dL, error_threshold, run_type=None):
    # get the pdf and cdf
    total_num, pdf_data = psd_2_pdf(data_bins)
    cdf_data = cdf_func(pdf_data)
    psd_data = pdf_2_psd(pdf_data, total_num)
    # search for the valid region
    start, middle, end = search_for_valid_region(cdf_data, valid_lower_bound, valid_upper_bound)
    # place the observation points
    ob_points_id, middle_id, width = place_ob_points(start, end, n_ob_points, dL)
    # get the real particle distribution of the observation points
    ob_dist = real_dist_ob(psd_data, ob_points_id)
    # predict the psd
    y_pre = predict_psd_extrapolate(psd_data, ob_points_id, ob_dist)
    # calculate the error
    error = error_extrapolate(psd_data, y_pre)
    # check run type to decide whether run the following code
    if run_type == 'optimal':
        if np.all(error < error_threshold):
            print('The sparse model is good enough')
            # plot_sparse(0.5, ob_points_id, ob_dist, x, data_bins, y_pre)
            alert = False
        else:
            print('More observation points are needed')
            plot_sparse(0.5, ob_points_id, ob_dist, x, data_bins, y_pre)
            # signal the user to add more observation points
            alert = True
    else:
        alert = False

    return alert, ob_dist, ob_points_id, middle_id, width, y_pre


def find_optimal_ob_points(data, valid_lower_bound, valid_upper_bound, n_points, dL, error_threshold):
    """
    find the optimal number of observation points
    :param data:
    :param valid_lower_bound:
    :param valid_upper_bound:
    :param n_points:
    :param dL:
    :param error_threshold:
    :return:
    """
    alert = sparse_model(data, valid_lower_bound, valid_upper_bound, n_points, dL,
                         error_threshold=error_threshold, run_type='optimal')[0]
    while alert:  # and n_points < 80:
        n_points += 2
        alert = sparse_model(data, valid_lower_bound, valid_upper_bound, n_points, dL,
                             error_threshold=error_threshold, run_type='optimal')[0]

    return n_points


def main(save_name, valid_lower_bound, valid_upper_bound, input_n_ob_points, output_n_ob_points, dL,
         error_threshold):
    # load the data
    # input matrix
    # file_path_in = 'data/PBE_inputMatrix/'
    file_path_in = 'D:/PycharmProjects/surrogatepbe/PBEsolver_InputMatrix/'
    data_input = pd.read_csv(file_path_in + f"{save_name}.csv")
    data_input_original = data_input.copy(deep=True)
    for i in data_input.columns:
        if 'pop_bin' not in i:
            data_input.drop(i, axis=1, inplace=True)

    print('input data has been loaded')

    # find the optimal number of observation points for input matrix
    input_n_ob_points = find_optimal_ob_points(data_input,
                                               valid_lower_bound,
                                               valid_upper_bound,
                                               input_n_ob_points,
                                               dL,
                                               error_threshold=error_threshold)
    print('the optimal number of observation points for input matrix is\n', input_n_ob_points)

    # get the necessary observation info and combine it to previous info
    input_alert, input_observe_points, input_observe_points_id, input_middle_id, input_width, input_pred \
        = sparse_model(data_input, valid_lower_bound, valid_upper_bound, input_n_ob_points, dL, error_threshold)
    input_ob_columns = [f"ob_{x}" for x in range(input_n_ob_points)] + ['width'] + ['middle']
    input_observe_df = pd.DataFrame(
        np.concatenate((input_observe_points, input_width.reshape(-1, 1), input_middle_id.reshape(-1, 1)), axis=1),
        columns=input_ob_columns)
    input_mat = pd.concat([data_input_original, input_observe_df], axis=1)

    print('the input matrix has been generated')

    # output matrix
    # find the optimal number of observation points for output matrix
    # file_path_out = 'data/PBE_outputs/'
    file_path_out = 'D:/PycharmProjects/surrogatepbe/PBEsolver_outputs/'
    for runID in data_input_original["runID"]:
        print(f"runID {int(runID)}")
        try:
            file = pd.read_csv(file_path_out + f"PBEsolver_{save_name}_runID{int(runID)}.csv")
            for m in file.columns:
                if 'pop_bin' not in m:
                    file.drop(m, axis=1, inplace=True)

            output_n_ob_points = find_optimal_ob_points(file,
                                                        valid_lower_bound,
                                                        valid_upper_bound,
                                                        output_n_ob_points,
                                                        dL,
                                                        error_threshold=error_threshold)
        except:
            pass

    print('the optimal number of observation points for output matrix is\n', output_n_ob_points)


    output_mat = {}
    # unreliable_runID = []
    print('generating output matrix')
    for k in data_input_original["runID"]:
        print(f"runID {int(k)}")
        file = pd.read_csv(file_path_out + f"PBEsolver_{save_name}_runID{int(k)}.csv")
        output_others = file[['c', 't']]
        for m in file.columns:
            if 'pop_bin' not in m:
                file.drop(m, axis=1, inplace=True)

        output_alert, output_observe_points, output_observe_points_id, output_middle_id, output_width, output_pred \
            = sparse_model(file, valid_lower_bound, valid_upper_bound, output_n_ob_points, dL, error_threshold)
        output_ob_columns = [f"ob_{x}" for x in range(output_n_ob_points)] + ['width'] + ['middle']
        output_observe_df = pd.DataFrame(
            np.concatenate((output_observe_points, output_width.reshape(-1, 1), output_middle_id.reshape(-1, 1)),
                           axis=1), columns=output_ob_columns)
        output_mat[k] = pd.concat([output_others, output_observe_df], axis=1)

    # generate training data
    X, Y = reformat_input_output(input_mat, output_mat, input_n_ob_points, output_n_ob_points, t_sample_frac=0.25,
                                 no_sims=5000, shuffle=False)

    # save the data as csv file
    export_path = 'data/sparse_training_data/'
    export_name = f"{save_name}_input_{input_n_ob_points}_{output_n_ob_points}.csv"
    X_df = pd.DataFrame(X)
    X_df.to_csv(export_path + export_name, index=True)
    export_name = f"{save_name}_output_{input_n_ob_points}_{output_n_ob_points}.csv"
    Y_df = pd.DataFrame(Y)
    Y_df.to_csv(export_path + export_name, index=True)

    # return X, Y, unreliable_runID
    return X, Y


def plot_sparse(x, data, y_pred):
    # the scatter plot of observation points
    plt.figure(figsize=(6, 8))
    plt.subplot(2, 1, 1)
    # plt.scatter(x_id_matrix[0, :], observe_points[0, :], label="ob_points", color="r")
    plt.plot(x, data.iloc[0, :], label="original", color="b")
    plt.plot(x, y_pred[0, :], label="estimated", color="r")
    plt.legend(prop={'size': 15})
    # set the legend size
    # plt.legend(prop={'size': 8})
    # plt.ylabel(r"PSD $f$ [m$^{-3}\mu$m$^{-1}$]")

    plt.subplot(2, 1, 2)
    # plt.scatter(x_id_matrix[-1, :], observe_points[-1, :], label="ob_points", color="r")
    plt.plot(x, data.iloc[-1, :], label="original", color="b")
    plt.plot(x, y_pred[-1, :], label="estimated", color="r")
    plt.legend(prop={'size': 15})

    plt.show()


if __name__ == '__main__':
    # # -----------------case 1: test the sparse model-----------------
    # load x
    L_max = 500  # [um]
    dL = 0.5  # [um]
    L_bounds = np.arange(0, L_max + dL, dL)  # [um]
    L_mid = np.mean([L_bounds[:-1], L_bounds[1:]], axis=0)  # [um]
    x = L_mid
    #
    # # load cdf
    # filepath = 'data/'
    # file_name = 'PBEsolver_InputMat_231015_2234_runID0.csv'
    # data = pd.read_csv(filepath + file_name)
    # for i in data.columns:
    #     if 'pop_bin' not in i:
    #         data.drop(i, axis=1, inplace=True)
    #
    # # test the sparse model
    # valid_lower_bound = 0.001
    # valid_upper_bound = 0.999
    # n_ob_points = 3
    # dL = 0.5
    # alert, observe_points, observe_points_id, middle_id, width, y_pred = sparse_model(data,
    #                                                                                   valid_lower_bound,
    #                                                                                   valid_upper_bound,
    #                                                                                   n_ob_points,
    #                                                                                   dL,
    #                                                                                   error_threshold=0.0001)
    #
    # # plot the result
    # plot_sparse(dL, observe_points_id, observe_points, x, data, y_pred)
    #
    # # find the optimal number of observation points
    # while alert:
    #     n_ob_points += 2
    #     print('the number of observation points is\n', n_ob_points)
    #     alert, observe_points, observe_points_id, middle_id, width, y_pred = sparse_model(data,
    #                                                                                       valid_lower_bound,
    #                                                                                       valid_upper_bound,
    #                                                                                       n_ob_points,
    #                                                                                       dL,
    #                                                                                       error_threshold=0.0001)
    # print('the final number of observation points is\n', n_ob_points)
    #
    # # plot the result
    # plot_sparse(dL, observe_points_id, observe_points, x, data, y_pred)

    # -----------------case 2: run the whole pipeline-----------------
    # 1207_1605,53, 93
    # 1213_1132, 53, 91
    X, Y = main(save_name='InputMat_231213_1132',
                valid_lower_bound=0.001,
                valid_upper_bound=0.999,
                input_n_ob_points=91,
                output_n_ob_points=91,
                dL=0.5,
                error_threshold=0.05)
