import logging
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.optimize import curve_fit

from .model import MtanClassif
from .data_loader import load_df, TimeSeriesDataset
from utils import get_path, load_json

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s:%(lineno)d:%(levelname)s:%(name)s:%(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

green = '#b30000'
blue = '#4421af'
purple = '#5ad45a'


def approx_pyranometer():
    # Data
    x = list(range(1, 25))
    y = [-0.6911154657947024, -0.6907105136062563, -0.6907625775035586, -0.6904118897412882,
         -0.6903448002560898, -0.6900468860713898, -0.6732346719240069, -0.5607986153270463,
         -0.151974961928455, 0.5479398509710395, 1.1191689976716854, 1.514523286453948,
         1.635221972000354, 1.5521605894304145, 1.358338337921811, 0.9992461909025483,
         0.49544929489072237, 0.011698060586847877, -0.36533960200204496, -0.6074110011760002,
         -0.6848462475705492, -0.6926189840557192, -0.6917751588375416, -0.6913123365887286]

    x_data = np.array(x)
    y_data = np.array(y)

    # Define the sine function to fit
    def sine_function(x, A, B, C, D):
        return A * np.sin(B * x + C) + D

    # Initial guess for the sine function parameters
    initial_guess_sine = [1, 0.2, 0, 0]

    # Perform the curve fitting for the sine function
    params_sine, _ = curve_fit(sine_function, x_data, y_data, p0=initial_guess_sine)

    plt.plot(x_data, y_data, 'o', label='True data-points', color=green)

    y_data[7] = -0.69

    for index_start, index_end in [(7, 13), (13, 21)]:
        # Define the indices
        y_start, y_end = y_data[index_start], y_data[index_end]

        # Create the linear line (linear interpolation)
        x_line = np.arange(index_start, index_end + 1)
        y_line = np.linspace(y_start, y_end, len(x_line))

        # Plot the linear interpolation line
        plt.plot(x_line, y_line, '-', color=blue)

    x_line = np.arange(1, 8)
    y_line = np.full(7, -0.69)
    plt.plot(x_line, y_line, '-', color=blue)

    x_line = np.arange(21, 25)
    y_line = np.full(4, -0.69)
    plt.plot(x_line, y_line, '-', label='Approx. Triangular Pulse', color=blue)

    for x in x_data:
        plt.axvline(x=x, color='gray', alpha=0.2)

    x_smooth = np.linspace(0, 23, 1000)
    y_smooth = sine_function(x_smooth, *params_sine)
    plt.plot(x_smooth, y_smooth, '-', label='Approx. Sine Function', color=purple)
    plt.xlabel('Hour in the day')
    plt.ylabel('Mean Pyranometer')
    plt.legend(fontsize='small')

    cfn = get_path(dirs=["models"], name=f"fit_pyranometer.png")
    plt.savefig(cfn, dpi=400, bbox_inches='tight')
    plt.clf()


def evaluate_time_repr_lin(model, dataloader, seed, target_layer, pred_value, time_representation, model_name):
    model.eval()

    tensor_list = []

    def hook_fn(module, input, output):
        tensor_list.append(output)

    hook_handle = target_layer.register_forward_hook(hook_fn)

    for (X, masks_X, observed_tp), (y, mask_y) in dataloader:
        X, masks_X, y, mask_y = X.to(device), masks_X.to(device), y.to(device), mask_y.to(device)

        with torch.no_grad():
            _ = model(X, observed_tp, masks_X)

    stacked_tensor = torch.cat(tensor_list, dim=0).squeeze()
    mean_tensor = torch.mean(stacked_tensor, dim=0).squeeze()
    std_tensor = torch.std(stacked_tensor, dim=0).squeeze()

    std_array = std_tensor.cpu().numpy()
    day_array = mean_tensor.cpu().numpy()
    x_values = range(1, 25)

    upper_bound = std_array + day_array
    lower_bound = -std_array + day_array

    # Plot the data along with the upper and lower bounds
    plt.plot(x_values, upper_bound, linestyle='--', color=green, alpha=0.3, label='Upper Bound')
    plt.plot(x_values, lower_bound, linestyle='--', color=green, alpha=0.3, label='Lower Bound')
    plt.fill_between(x_values, upper_bound, lower_bound, color=green, alpha=0.3)

    for x in x_values:
        plt.axvline(x=x, color='gray', alpha=0.3)

    # Plot the data
    plt.plot(x_values, day_array, marker='o', color=blue)
    plt.xlabel('Hour in the day')
    plt.ylabel('Time representation values')

    cfn = get_path(dirs=["models", time_representation, pred_value, "mTAN", "best_model", "time_repr"],
                   name=f"{model_name}_feature_linear.png")
    plt.savefig(cfn, dpi=400, bbox_inches='tight')
    plt.clf()
    '''

    stacked_tensor = torch.cat(tensor_list, dim=0).squeeze()

    j = 'linear'
    x_values = range(1, 25)

    for x in x_values:
        plt.axvline(x=x, color='gray', alpha=0.3)

    for i in [0, 3, 24, 29]:
        day_array = stacked_tensor[i].cpu().numpy()
        # Plot the data
        plt.scatter(x_values, day_array, marker='o', label=i)

    plt.xlabel('Hour in the day')
    plt.ylabel('Time representation values')
    plt.title(f'Plot of mean time feature {j}')
    plt.legend()

    cfn = get_path(dirs=["models", time_representation, pred_value, "mTAN", str(seed), "time_repr"], name=f"feature_{j}.png")
    # plt.savefig(cfn, dpi=400)
    plt.show()
    plt.clf()
    '''

    hook_handle.remove()


def evaluate_time_repr_sin(model, dataloader, seed, target_layer, pred_value, time_representation, model_name):
    model.eval()

    tensor_list = []

    def hook_fn(module, input, output):
        tensor_list.append(torch.sin(output))

    hook_handle = target_layer.register_forward_hook(hook_fn)

    for (X, masks_X, observed_tp), (y, mask_y) in dataloader:
        X, masks_X, y, mask_y = X.to(device), masks_X.to(device), y.to(device), mask_y.to(device)

        with torch.no_grad():
            _ = model(X, observed_tp, masks_X)

    '''flattened_list = [tensor.squeeze().transpose(0, 1).view(-1).cpu().numpy() for tensor in tensor_list]
    df = pd.DataFrame(flattened_list)
    cfn = get_path(dirs=["models", time_representation, pred_value, "mTAN", "best_model"],
                   name=f"sin_time_representation.csv")
    df.to_csv(cfn, index=False)'''

    stacked_tensor = torch.cat(tensor_list, dim=0)
    mean_tensor = torch.mean(stacked_tensor, dim=0)
    std_tensor = torch.std(stacked_tensor, dim=0)

    model_epoch = "Final model" if len("mTAN.pth") == len(model_name) \
        else f"Epoch {model_name[len('mTAN_'):len(model_name) - len('.pth')]}"

    for j in range(mean_tensor.shape[1]):
        std_ = std_tensor[:, j]
        std_array = std_.cpu().numpy()
        day_tensor = mean_tensor[:, j]
        day_array = day_tensor.cpu().numpy()
        x_values = range(1, 25)

        upper_bound = std_array + day_array
        lower_bound = -std_array + day_array

        # Plot the data along with the upper and lower bounds
        plt.plot(x_values, upper_bound, linestyle='--', color=green, alpha=0.2, label='Upper Bound')
        plt.plot(x_values, lower_bound, linestyle='--', color=green, alpha=0.2, label='Lower Bound')
        plt.fill_between(x_values, upper_bound, lower_bound, color=green, alpha=0.2)

        for x in x_values:
            plt.axvline(x=x, color='gray', alpha=0.2)

        # Plot the data
        plt.plot(x_values, day_array, marker='o', color=blue)
        plt.xlabel('Hour in the day')
        plt.ylabel('Time representation values')

        cfn = get_path(dirs=["models", time_representation, pred_value, "mTAN", "best_model", "time_repr"],
                       name=f"{model_name}_feature_{j}.png")
        plt.savefig(cfn, dpi=400, bbox_inches='tight')
        plt.clf()

    # Plot sin regarding the groups
    groups = [
        [0, 2, 5, 7, 8, 15, 17],
        [1, 3, 9, 10, 12, 13, 18, 20, 22],
        [4, 6, 19, 21],
        [11],
        [14],
        [16]]

    x_values = range(1, 25)

    for grp_num, group in enumerate(groups):

        for x in x_values:
            plt.axvline(x=x, color='gray', alpha=0.2)

        for j in group:
            std_ = std_tensor[:, j]
            std_array = std_.cpu().numpy()
            day_tensor = mean_tensor[:, j]
            day_array = day_tensor.cpu().numpy()

            upper_bound = std_array + day_array
            lower_bound = -std_array + day_array

            # Plot the data along with the upper and lower bounds
            plt.plot(x_values, upper_bound, linestyle='--', color=green, alpha=0.1, label='Upper Bound')
            plt.plot(x_values, lower_bound, linestyle='--', color=green, alpha=0.1, label='Lower Bound')
            plt.fill_between(x_values, upper_bound, lower_bound, color=green, alpha=0.1)

            # Plot the data
            plt.plot(x_values, day_array, marker='o')

        plt.xlabel('Hour in the day')
        plt.ylabel('Time representation values')

        cfn = get_path(dirs=["models", time_representation, pred_value, "mTAN", "best_model", "grouped_time_repr"],
                       name=f"{model_name}_feature_group_{grp_num}.png")
        plt.savefig(cfn, dpi=400, bbox_inches='tight')
        plt.clf()

    hook_handle.remove()


def evaluate_time_repr_pulse(model, dataloader, seed, target_layer, pred_value, time_representation, model_name):
    model.eval()

    tensor_list = []

    def triangular_pulse(linear_tensor):
        max_pos = 11
        distance = torch.abs(linear_tensor - linear_tensor[:, max_pos, :].unsqueeze(1))
        a = torch.quantile(distance, 0.25)
        # Calculate the triangular pulse values
        condition = distance <= a
        pulse = torch.where(condition, 1 - (distance / a), torch.tensor(0.0))
        return pulse

    def hook_fn(module, input, output):
        tensor_list.append(triangular_pulse(output))

    hook_handle = target_layer.register_forward_hook(hook_fn)

    for (X, masks_X, observed_tp), (y, mask_y) in dataloader:
        X, masks_X, y, mask_y = X.to(device), masks_X.to(device), y.to(device), mask_y.to(device)

        with torch.no_grad():
            _ = model(X, observed_tp, masks_X)

    pulse = torch.cat(tensor_list, dim=0)

    x_values = range(1, 25)

    for x in x_values:
        plt.axvline(x=x, color='gray', alpha=0.3)

    day_tensor = pulse[10, :, 0]
    day_array = day_tensor.detach().cpu().numpy()
    plt.plot(x_values, day_array, marker='o')

    day_tensor = pulse[10, :, 5]
    day_array = day_tensor.detach().cpu().numpy()
    plt.plot(x_values, day_array, marker='o')

    day_tensor = pulse[10, :, 8]
    day_array = day_tensor.detach().cpu().numpy()
    plt.plot(x_values, day_array, marker='o')

    day_tensor = pulse[10, :, 10]
    day_array = day_tensor.detach().cpu().numpy()
    plt.plot(x_values, day_array, marker='o')

    day_tensor = pulse[10, :, 20]
    day_array = day_tensor.detach().cpu().numpy()
    plt.plot(x_values, day_array, marker='o')

    model_epoch = "Final model" if len("mTAN.pth") == len(model_name) \
        else f"Epoch {model_name[len('mTAN_'):len(model_name) - len('.pth')]}"

    plt.xlabel('Hour in the day')
    plt.ylabel('Time representation values')

    cfn = get_path(dirs=["models", time_representation, pred_value, "mTAN", "best_model", "time_repr"],
                   name=f"{model_name}_feature.png")
    plt.savefig(cfn, dpi=400, bbox_inches='tight')
    plt.clf()

    hook_handle.remove()


def test_model_time_repr(X_cols, y_cols, params, sequence_length, seed, time_representation):
    pvt_cols = ["DATETIME"] + X_cols + y_cols
    pred_value = y_cols[0]
    dim = len(X_cols)

    # Parameters:
    num_heads = params["num_heads"]
    embed_time = params["embed_time"]

    mean_stds = load_json('mTAN/mean_stds.json')
    test_df, _ = load_df(df_path="../../data/test_set_noa_classes.csv", pvt_cols=pvt_cols, parse_dates=["DATETIME"],
                         normalize=True,
                         stats=mean_stds, y_cols=y_cols)

    test_df.to_csv("../../data/pvt_df_test.csv")
    test_df = pd.read_csv("../../data/pvt_df_test.csv", parse_dates=['DATETIME'], index_col='DATETIME')

    # Create a dataset and dataloader
    testing_dataset_per_day = TimeSeriesDataset(dataframe=test_df, sequence_length=sequence_length, X_cols=X_cols,
                                                y_cols=y_cols, per_day=True)

    test_loader_per_day = DataLoader(testing_dataset_per_day, batch_size=1, shuffle=False, drop_last=False)

    trained_model = MtanClassif(input_dim=dim, query=torch.linspace(0, 1., embed_time), embed_time=embed_time,
                                num_heads=num_heads, device=device, time_representation=time_representation).to(device)

    models_to_test = ["mTAN_1.pth", "mTAN.pth"]

    for model_name in models_to_test:

        mfn = get_path(dirs=["models", time_representation, pred_value, "mTAN", str(seed)], name=model_name)
        trained_model.load_state_dict(torch.load(mfn))

        if time_representation == "sin":
            target_layer = trained_model.periodic
            evaluate_time_repr_sin(model=trained_model, dataloader=test_loader_per_day, target_layer=target_layer,
                                   seed=seed, pred_value=pred_value, time_representation=time_representation,
                                   model_name=model_name)
        else:
            target_layer = trained_model.periodic
            evaluate_time_repr_pulse(model=trained_model, dataloader=test_loader_per_day, target_layer=target_layer,
                                     seed=seed, pred_value=pred_value, time_representation=time_representation,
                                     model_name=model_name)

        target_layer = trained_model.linear
        evaluate_time_repr_lin(model=trained_model, dataloader=test_loader_per_day, target_layer=target_layer,
                               seed=seed, pred_value=pred_value, time_representation=time_representation,
                               model_name=model_name)


def main_loop_test_time_repr(seed, y_cols, time_representation):
    sequence_length = 24

    X_cols = ["PYRANOMETER", "OUTDOOR_TEMP"]
    params = {'num_heads': 8, 'embed_time': 24}

    test_model_time_repr(X_cols=X_cols, y_cols=y_cols, params=params, sequence_length=sequence_length, seed=seed,
                         time_representation=time_representation)

