import numpy as np
import pandas as pd
import scipy as sp
from base_functions import psd_2_pdf, cdf_func, error_functions


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
    # todo: check
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
        error[i] = error_functions(data_pdf.iloc[i, :], y_pre[i, :].reshape(-1, 1))
    return error


def sparse_model(data_bins, valid_lower_bound, valid_upper_bound, n_ob_points, dL):
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
    if error.all() < 0.01:
        print('the fitting performance is good')
        alert = False
    else:
        print('More observation points are needed')
        # signal the user to add more observation points
        alert = True
    return alert, middle_id, width


if __name__ == '__main__':
    # load x
    L_max = 500  # [um]
    dL = 0.5  # [um]
    L_bounds = np.arange(0, L_max + dL, dL)  # [um]
    L_mid = np.mean([L_bounds[:-1], L_bounds[1:]], axis=0)  # [um]
    x = L_mid

    # load cdf
    filepath = 'data/'
    file_name = 'PBEsolver_InputMat_231015_2234_runID1.csv'
    data = pd.read_csv(filepath + file_name)
    for i in data.columns:
        if 'pop_bin' not in i:
            data.drop(i, axis=1, inplace=True)

    # test the sparse model
    valid_lower_bound = 0.001
    valid_upper_bound = 0.999
    n_ob_points = 51
    dL = 0.5
    alert, middle_id, width = sparse_model(data, valid_lower_bound, valid_upper_bound, n_ob_points, dL)
    # find the optimal number of observation points
    while alert:
        n_ob_points += 2
        print('the number of observation points is\n', n_ob_points)
        alert, middle_id, width = sparse_model(data, valid_lower_bound, valid_upper_bound, n_ob_points, dL)
    print('the final number of observation points is\n', n_ob_points)

