import pickle

from sparse_preprocessing import *

"""
The main idea is that with the sparse model on the last time step of the simulation, we can use the fixed observation 
locations of the observation points.
Theoretically, with those ids, we can get the estimated distribution of each time step.
"""


def sparse_model_on_fixed_locations(bin_data, fixed_observe_locations, fixed_middle_id, fixed_width, error_threshold):
    """
    For all the distributions in one simulation, we use the same observation locations to estimate the distribution.
    :param bin_data: the data of the original bins
    :param fixed_observe_locations: vector, the fixed observation locations, determined by the last time step
    :param fixed_middle_id: vector of (1,), the middle id of the fixed observation locations
    :param fixed_width: vector of (1,), the width of the fixed observation locations
    :param error_threshold: the threshold of the error
    """
    # todo: in some cases, only growth
    # the logic could be:
    # 1. see if the nucleation parameters are zeros (only growth)
    # or the first observe location of the last time step is large (only growth)
    # 2. if yes, then the shape keeps 'constant', so the values observed is quite similar,
    # the only thing the model needs to learn is the middle and width
    # 3. if no, the nucleation and growth coexist in the system,
    # then the shape changes, so the values observed are different,
    # the model needs to learn the shape, middle and width, it is more difficult
    # so if we use the fixed observation locations, the model will be more robust
    # which is more like the de-resolutioned model with a more smaller window
    # todo: whether the model can learn these two patterns at the same time?

    # modify the fixed observation locations
    fixed_observe_locations = np.tile(fixed_observe_locations, (bin_data.shape[0], 1))

    # get the observation value
    ob_psd = real_dist_ob(bin_data, fixed_observe_locations)

    # predict the psd
    y_pre = predict_psd_extrapolate(bin_data, fixed_observe_locations, ob_psd)
    # calculate the error
    error = error_extrapolate(bin_data, y_pre)
    # check run type to decide whether run the following code
    if np.all(error < error_threshold):
        print('The sparse model is good enough')
        alert = False
    else:
        print('More observation points are needed')
        plot_sparse(x, bin_data, y_pre)
        # signal the user to add more observation points
        alert = True
    # replicate the fixed parameters
    ob_points_id = np.tile(fixed_observe_locations, (bin_data.shape[0], 1))
    middle_id = np.tile(fixed_middle_id, (bin_data.shape[0], 1))
    width = np.tile(fixed_width, (bin_data.shape[0], 1))

    return alert, ob_psd, ob_points_id, middle_id, width, y_pre



def run(save_name, valid_lower_bound, valid_upper_bound, input_n_ob_points, output_n_ob_points, dL,
         error_threshold):
    # load the data
    # input matrix
    # file_path_in = 'data/PBE_inputMatrix/'
    file_path_in = 'D:/PycharmProjects/surrogatepbe/PBEsolver_InputMatrix/'
    data_input = pd.read_csv(file_path_in + f"{save_name}.csv")
    data_input_original = data_input.copy(deep=True)
    for i in data_input.columns:
        if 'pop_bin' not in i:
            data_input = data_input.drop(i, axis=1)

    print('input data has been loaded')

    # find the optimal number of observation points for input matrix
    input_n_ob_points = find_optimal_ob_points(data_input,
                                               valid_lower_bound,
                                               valid_upper_bound,
                                               input_n_ob_points,
                                               dL,
                                               error_threshold=error_threshold)
    print('the optimal number of observation points for input matrix is\n', input_n_ob_points)

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
                    file = file.drop(m, axis=1)

            output_n_ob_points = find_optimal_ob_points(file,
                                                        valid_lower_bound,
                                                        valid_upper_bound,
                                                        output_n_ob_points,
                                                        dL,
                                                        error_threshold=error_threshold)
        except:
            pass

    print('the optimal number of observation points for output matrix is\n', output_n_ob_points)

    # check which number of observation points is larger
    if input_n_ob_points >= output_n_ob_points:
        n_ob_points = input_n_ob_points
    else:
        n_ob_points = output_n_ob_points

    output_mat = {}
    # unreliable_runID = []
    print('generating output matrix')
    both_runID = []
    only_growth_runID = []
    fixed_ob_locations = {}
    fixed_ob_middle_id = {}
    fixed_ob_width = {}
    for k in data_input_original["runID"]:
        print(f"runID {int(k)}")
        file = pd.read_csv(file_path_out + f"PBEsolver_{save_name}_runID{int(k)}.csv")
        output_others = file[['c', 't']]
        data_last_step = file[file['t'] == file['t'].max()]
        for i in data_last_step.columns:
            if 'pop_bin' not in i:
                data_last_step = data_last_step.drop(i, axis=1)

        for m in file.columns:
            if 'pop_bin' not in m:
                file = file.drop(m, axis=1)
        # get the observation info of the last time step
        output_alert_last, output_ob_points_last, output_ob_points_id_last, output_mid_id_last, output_width_last, output_pred_last \
            = sparse_model(data_last_step, valid_lower_bound, valid_upper_bound, n_ob_points, dL, error_threshold)

        if output_ob_points_id_last[0][0] < 2:
            # it indicates that the nucleation and growth coexist in the system
            both_runID.append(k)
            output_alert, output_observe_points, output_ob_points_id, output_middle_id, output_width, output_y_pre = \
                sparse_model_on_fixed_locations(file, output_ob_points_id_last, output_mid_id_last, output_width_last, error_threshold)
            if output_alert:
                print(f'the fixed observation data is not good enough for {k}')
            fixed_ob_locations[k] = output_ob_points_id_last
            fixed_ob_middle_id[k] = output_mid_id_last
            fixed_ob_width[k] = output_width_last
        else:
            # it indicates that only growth exists in the system
            only_growth_runID.append(k)
            output_alert, output_observe_points, output_ob_points_id, output_middle_id, output_width, output_y_pred \
                = sparse_model(file, valid_lower_bound, valid_upper_bound, n_ob_points, dL, error_threshold)

        output_ob_columns = [f"ob_{x}" for x in range(n_ob_points)] + ['width'] + ['middle']
        output_observe_df = pd.DataFrame(
            np.concatenate((output_observe_points, output_width.reshape(-1, 1), output_middle_id.reshape(-1, 1)),
                           axis=1), columns=output_ob_columns)
        output_mat[k] = pd.concat([output_others, output_observe_df], axis=1)

    print('the output matrix has been generated')

    # process the input matrix
    input_mat = pd.DataFrame()
    for k in range(data_input.shape[0]):
        if k in both_runID:
            input_alert, input_observe_points, input_observe_points_id, input_middle_id, input_width, input_pred \
                    = sparse_model_on_fixed_locations(data_input[k: k + 1],  # todo: check the index
                                                      fixed_ob_locations[k],
                                                      fixed_ob_middle_id[k],
                                                      fixed_ob_width[k],
                                                      error_threshold)
        else:
            input_alert, input_observe_points, input_observe_points_id, input_middle_id, input_width, input_pred \
                = sparse_model(data_input[k: k + 1],
                               valid_lower_bound,
                               valid_upper_bound,
                               n_ob_points,
                               dL,
                               error_threshold)

        input_ob_columns = [f"ob_{x}" for x in range(n_ob_points)] + ['width'] + ['middle']
        input_observe_df = pd.DataFrame(
            np.concatenate((input_observe_points, input_width.reshape(-1, 1), input_middle_id.reshape(-1, 1)), axis=1),
            columns=input_ob_columns)
        input_mat[k: k+1] = pd.concat([data_input_original[k: k+1], input_observe_df], axis=1)

    print('the input matrix has been generated')

    # generate training data
    X, Y = reformat_input_output(input_mat, output_mat, input_n_ob_points, output_n_ob_points, t_sample_frac=0.25,
                                 no_sims=5000, shuffle=False)

    # save the data as csv file
    export_path = 'data/sparse_training_data/'
    export_name = f"{save_name}_input_{n_ob_points}_{n_ob_points}_fixed.csv"
    X_df = pd.DataFrame(X)
    X_df.to_csv(export_path + export_name, index=True)
    export_name = f"{save_name}_output_{n_ob_points}_{n_ob_points}_fixed.csv"
    Y_df = pd.DataFrame(Y)
    Y_df.to_csv(export_path + export_name, index=True)

    return X, Y, both_runID


if __name__ == '__main__':
    # # -----------------case 1: test the sparse model-----------------
    # load x
    L_max = 500  # [um]
    dL = 0.5  # [um]
    L_bounds = np.arange(0, L_max + dL, dL)  # [um]
    L_mid = np.mean([L_bounds[:-1], L_bounds[1:]], axis=0)  # [um]
    x = L_mid

    # file_path_in = 'data/PBEsolver_outputs/'
    # file = 'PBEsolver_InputMat_231021_0805_runID1.csv'
    # data_input = pd.read_csv(file_path_in + file)
    # data_bin = data_input.copy(deep=True)
    # for i in data_bin.columns:
    #     if 'pop_bin' not in i:
    #         data_bin = data_bin.drop(i, axis=1)
    # data_last_step = data_input[data_input['t'] == data_input['t'].max()]
    # for i in data_last_step.columns:
    #     if 'pop_bin' not in i:
    #         data_last_step = data_last_step.drop(i, axis=1)
    #
    # alert_last, ob_dist_last, ob_points_id_last, middle_id_last, width_last, y_pre_last = \
    #     sparse_model(
    #         data_last_step,
    #         valid_lower_bound=0.001,
    #         valid_upper_bound=0.999,
    #         n_ob_points=91,
    #         dL=0.5,
    #         error_threshold=0.05)
    #
    # alert, ob_dist, ob_points_id, middle_id, width, y_pre = sparse_model_on_fixed_locations(data_bin, ob_points_id_last, middle_id_last, width_last, error_threshold=0.05)
    #
    # a = 1

    # # # -----------------case 2: run the whole pipeline-----------------
    save_name = 'InputMat_231207_1605'
    X, Y, both_ids = run(save_name=save_name,
                         valid_lower_bound=0.001,
                         valid_upper_bound=0.999,
                         input_n_ob_points=91,
                         output_n_ob_points=91,
                         dL=0.5,
                         error_threshold=0.05)
    print(both_ids)
    # save the list of both_ids
    with open(f'data/sparse_training_data/both_ids_{save_name}.pkl', 'wb') as f:
        pickle.dump(both_ids, f)
