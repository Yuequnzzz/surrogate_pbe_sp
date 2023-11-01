import numpy as np
import pandas as pd
import scipy as sp
import copy
from base_functions import psd_2_pdf, cdf_func, error_func_scaled, error_functions, error_func_relative
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


def real_prob_ob(pdf_data, ob_points_id):
    """
    get the real probability of the observation points
    :param pdf_data: the pdf data
    :param ob_points_id: the id of observation points
    :return: the real probability of the observation points
    """
    # initialize
    ob_prob = np.zeros(ob_points_id.shape)
    # get the real probability by interpolation
    for i in range(ob_points_id.shape[0]):
        ob_prob[i, :] = np.interp(ob_points_id[i, :], np.arange(pdf_data.shape[1]), pdf_data.iloc[i, :].tolist())
    return ob_prob


def predict_pdf_extrapolate(pdf_data, ob_points_id, ob_prob):
    """
    Estimate how good the fitting performance is by calculating the error
    :param pdf_data: the pdf data
    :param ob_points_id: the id of observation points
    :param ob_prob: the real probability of the observation points
    :return: the predicted pdf
    """
    # initialize the error
    y_pre = np.zeros(pdf_data.shape)
    # use the observation points via interpolating to calculate the predicted probability at the original position
    for i in range(ob_points_id.shape[0]):
        # todo: in the later version, we need to consider 2d/3d extrapolation
        y_pre[i, :] = sp.interpolate.interp1d(ob_points_id[i, :], ob_prob[i, :],
                                              fill_value='extrapolate')(np.arange(pdf_data.shape[1]))
        # replace the negative value with 0
        y_pre[i, y_pre[i, :] < 0] = 0
    return y_pre


def error_extrapolate(data_pdf, y_pre):
    """
    Estimate how good the fitting performance is by calculating the error
    :param data_pdf: the pdf data
    :param y_pre: the predicted pdf
    :return: the error
    """
    # initialize the error
    error = np.zeros(data_pdf.shape[0])
    # calculate the error
    for i in range(data_pdf.shape[0]):
        # error[i] = error_func_relative(np.asarray(data_pdf.iloc[i, :]), y_pre[i, :].reshape(-1, 1))
        # todo: think of a better way to calculate the error
        error[i] = error_functions(data_pdf.iloc[i, :], y_pre[i, :].reshape(-1, 1))
    return error


def sparse_model(data_bins, valid_lower_bound, valid_upper_bound, n_ob_points, dL, error_threshold=0.01):
    # get the pdf and cdf
    pdf_data = psd_2_pdf(data_bins)
    cdf_data = cdf_func(pdf_data)
    # search for the valid region
    start, middle, end = search_for_valid_region(cdf_data, valid_lower_bound, valid_upper_bound)
    # place the observation points
    ob_points_id, middle_id, width = place_ob_points(start, end, n_ob_points, dL)
    # get the real probability of the observation points
    ob_prob = real_prob_ob(pdf_data, ob_points_id)
    # predict the pdf
    y_pre = predict_pdf_extrapolate(pdf_data, ob_points_id, ob_prob)
    # calculate the error
    error = error_extrapolate(pdf_data, y_pre)
    if np.all(error < error_threshold):
        print('the fitting performance is good')
        alert = False
    else:
        print('More observation points are needed')
        # signal the user to add more observation points
        alert = True
    return alert, ob_prob, ob_points_id, middle_id, width


def main(save_name, valid_lower_bound, valid_upper_bound, input_n_ob_points, output_n_ob_points, dL, error_threshold=0.01):
    # load the data
    # input matrix
    file_path_in = 'data/'
    data_input = pd.read_csv(file_path_in + f"{save_name}.csv")
    data_input_original = data_input.copy(deep=True)
    for i in data_input.columns:
        if 'pop_bin' not in i:
            data_input.drop(i, axis=1, inplace=True)

    # output matrix
    file_path_out = 'data/'
    results = {}
    for runID in data_input_original["runID"]:
        try:
            results[runID] = pd.read_csv(file_path_out + f"PBEsolver_{save_name}_runID{int(runID)}.csv")
        except:
            pass
    data_output_original = copy.deepcopy(results)
    for j in results.keys():
        if 'pop_bin' not in results[j].columns:
            results[j].drop(results[j].columns[0], axis=1, inplace=True)

    # get the necessary observation info and combine it to previous info
    input_alert, input_observe_points, input_observe_points_id, input_middle_id, input_width \
        = sparse_model(data_input, valid_lower_bound, valid_upper_bound, input_n_ob_points, dL)
    input_ob_columns = [f"ob_{x}" for x in range(input_n_ob_points)] + ['width'] + ['middle']
    input_observe_df = pd.DataFrame(np.concatenate((input_observe_points, input_width.reshape(-1, 1), input_middle_id.reshape(-1, 1)), axis=1), columns=input_ob_columns)
    input_mat = pd.concat([data_input_original, input_observe_df], axis=1)

    # todo: the output observation info is not correct
    for k in results.keys():

        output_alert, output_observe_points, output_observe_points_id, output_middle_id, output_width \
            = sparse_model(results[k], valid_lower_bound, valid_upper_bound, output_n_ob_points, dL)
        output_ob_columns = [f"ob_{x}" for x in range(output_n_ob_points)] + ['width'] + ['middle']
        output_observe_df = pd.DataFrame(np.concatenate((output_observe_points, output_width.reshape(-1, 1), output_middle_id.reshape(-1, 1)), axis=1), columns=output_ob_columns)
        results[k] = pd.concat([data_output_original[k], output_observe_df], axis=1)




    input_columns = ['runID', 'T0', 'dT', 'dt', 'S0', 'sol_k0', 'sol_kT', 'growth_k0', 'growth_kS',
                     'nuc_k0', 'nuc_kS', 'ini_mu0']
    output_columns = ["c"]
    #
    # X, Y = [], []
    # for runID, res in results.items():
    #     res = res.sample(frac=t_sample_frac)
    #
    #     no_timepoints = res.shape[0]
    #     Y.append(np.array(res[output_columns]))
    #
    #     relevant_inputs = np.array(input_mat.query("runID == @runID")[input_columns])
    #     relevant_inputs_repeated = np.vstack([relevant_inputs] * no_timepoints)
    #
    #     t_vec = np.array(res["t"])[..., np.newaxis]
    #     x = np.hstack([t_vec, relevant_inputs_repeated])
    #
    #     X.append(x)
    #     if len(X) > no_sims:
    #         break
    #
    # X = np.vstack(X)
    # Y = np.vstack(Y)
    #
    # if shuffle:
    #     ix = np.random.permutation(X.shape[0])
    #     X = X[ix, :]
    #     Y = Y[ix, :]
    #
    # print("X, Y dimensions: ", X.shape, Y.shape)
    return




if __name__ == '__main__':
    # # load x
    # L_max = 500  # [um]
    # dL = 0.5  # [um]
    # L_bounds = np.arange(0, L_max + dL, dL)  # [um]
    # L_mid = np.mean([L_bounds[:-1], L_bounds[1:]], axis=0)  # [um]
    # x = L_mid
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
    # n_ob_points = 41
    # dL = 0.5
    # alert, observe_points, observe_points_id, middle_id, width = sparse_model(data, valid_lower_bound, valid_upper_bound, n_ob_points, dL)
    #
    # # the scatter plot of observation points
    # plt.figure(figsize=(6, 8))
    # x_id_matrix = dL * observe_points_id
    # plt.subplot(2, 1, 1)
    # plt.scatter(x_id_matrix[0, :], observe_points[0, :], label="start", color="b")
    # plt.plot(x, psd_2_pdf(data).iloc[0, :], label="start", color="b")
    # plt.legend()
    # plt.ylabel(r"PSD $f$ [m$^{-3}\mu$m$^{-1}$]")
    #
    # plt.subplot(2, 1, 2)
    # plt.scatter(x_id_matrix[-1, :], observe_points[-1, :], label="end", color="r")
    # plt.plot(x, psd_2_pdf(data).iloc[-1, :], label="end", color="r")
    # plt.legend()
    #
    # plt.show()
    #
    # # find the optimal number of observation points
    # while alert:
    #     n_ob_points += 2
    #     print('the number of observation points is\n', n_ob_points)
    #     alert, observe_points, middle_id, width = sparse_model(data,
    #                                                            valid_lower_bound,
    #                                                            valid_upper_bound,
    #                                                            n_ob_points,
    #                                                            dL,
    #                                                            error_threshold=0.001)
    # print('the final number of observation points is\n', n_ob_points)
    main(save_name='InputMat_231015_2234', valid_lower_bound=0.001, valid_upper_bound=0.999, input_n_ob_points=41,
         output_n_ob_points=35, dL=0.5, error_threshold=0.01)


