import time
import json
import numpy as np
import yaml

from transformer.train import main_loop as train_transformer
from mTAN.train_and_test_classif import main_loop_train as train_mTAN


def train_models_fixed():
  
    with open("../config.yaml", "r") as config:
            config = yaml.safe_load(config)

    time_reprs = config["time_reprs"]
    seeds = config["seeds"]
    bins = config["bins"]

    models = ["transformer"]

    train_times = {}
    for _, time_repr_value in time_reprs.items():
        time_repr_str = time_repr_value['cors'][0] + '_and_' + time_repr_value['uniqs'][0]
        train_times[time_repr_str] = {}
        for bin, _ in bins:
            train_times[time_repr_str][bin] = {"transformer": {}}

    model_train = {"transformer": train_transformer}

    # Start training the models for each time_repr, seed and binning.
    # The training information is stored within the folder 'models/{time_repr_str}/{bin}/{model_name}'
    for _, time_repr_value in time_reprs.items():
        for bin, _ in bins:
            for seed in seeds:
                for model in models:
                    time_repr_str = time_repr_value['cors'][0] + '_and_' + time_repr_value['uniqs'][0]
                    print(f'Start training time_repr={time_repr_str}, bin={bin} with model "{model}" for seed={seed}')
                    start_time = time.time()

                    time_repr = (time_repr_value['dts'], time_repr_value['cors'], time_repr_value['uniqs'], [[tuple(arg) for arg in args_list] for args_list in time_repr_value['args']])
                    dirs = ["models", str(time_repr_str), str(bin), model, str(seed)]
                    model_train[model](time_repr, seed, [bin], dirs)

                    # Store training time for this model
                    train_times[time_repr_str][bin][model][seed] = time.time() - start_time
                    print(f'End training time_repr={time_repr_str}, bin={bin} with model "{model}" for seed={seed} [training time:{train_times[time_repr_str][bin][model][seed]:.2f}]')

    # Calculate mean and std of training statistics for each model
    for _, time_repr_value in time_reprs.items():
        for bin, _ in bins:
            time_repr_str = time_repr_value['cors'][0] + '_and_' + time_repr_value['uniqs'][0]
            bin_folder_path = f'models/{time_repr_str}/{bin}'
            acc_train_stats = {"transformer": dict()}
            for model in models:
                epoch_time = []
                folder_path = f'{bin_folder_path}/{model}'

            for s in seeds:
                seed = str(s)
                with open(f'{folder_path}/{seed}/train_checkpoints.json', "r") as file:
                    checkpoint_data = json.load(file)
                    # Training time per epoch
                    epoch_time += [train_times[time_repr_str][bin][model][s] / checkpoint_data["epochs"]]

                    for key in checkpoint_data:
                        if key not in ["seed"]:
                            try: acc_train_stats[model][key] += [checkpoint_data[key]]
                            except KeyError: acc_train_stats[model][key] = [checkpoint_data[key]]

            # Accumulate mean and std of training statistics
            keys = [key for key in acc_train_stats[model]]
            for key in keys:
                acc_train_stats[model][f'mean_{key}'] = np.mean(acc_train_stats[model][key])
                acc_train_stats[model][f'std_{key}'] = np.std(acc_train_stats[model][key])

            # Mean and std of training time per epoch
            acc_train_stats[model]["mean_epoch_time"] = np.mean(epoch_time)
            acc_train_stats[model]["std_epoch_time"] = np.std(epoch_time)

            # Keep only statistics
            acc_train_stats[model] = {key: value for key, value in acc_train_stats[model].items()
                                        if key.startswith("mean") or key.startswith("std")}

            # Write the dictionary to a JSON file
            with open(f'{folder_path}/acc_train_stats.json', "w") as file:
                json.dump(acc_train_stats[model], file)

                       
def train_models_learned():
  
    with open("../config.yaml", "r") as config:
        config = yaml.safe_load(config)

    seeds = config["seeds"]
    bins = config["bins"]
    time_representations = config["time_repr"]

    model = "mTAN"

    train_times = {}
    for bin, _ in bins:
        train_times[bin] = {"mTAN": dict()}

    model_train = {"mTAN": train_mTAN}

    # Start training the models for each seed and binning.
    # The training information is stored within the folder 'models/{bin}/{model_name}'
    for bin, weights in bins:
        for seed in seeds:
            for time_representation in time_representations:
                print(f'Start training bin={bin} with model "{model}" for seed={seed} and '
                      f'time representation={time_representation}')
                start_time = time.time()
                model_train[model](seed, [bin], weights, time_representation)
                # Store training time for this model
                train_times[bin][model][seed] = time.time() - start_time
                print(f'End training bin={bin} with model "{model}" for seed={seed} and '
                      f'time representation={time_representation} [training time:{train_times[bin][model][seed]:.2f}]')

    # Calculate mean and std of training statistics for each model
    for bin, _ in bins:
        for time_representation in time_representations:
            bin_folder_path = f'models/{time_representation}/{bin}'
            acc_train_stats = {"mTAN": dict()}

            epoch_time = []
            folder_path = f'{bin_folder_path}/{model}'

            for s in seeds:
                seed = str(s)
                with open(f'{folder_path}/{seed}/train_checkpoints.json', "r") as file:
                    checkpoint_data = json.load(file)
                    # Training time per epoch
                    epoch_time += [train_times[bin][model][s] / checkpoint_data["epochs"]]

                    for key in checkpoint_data:
                        if key not in ["seed"]:
                            try: acc_train_stats[model][key] += [checkpoint_data[key]]
                            except KeyError: acc_train_stats[model][key] = [checkpoint_data[key]]

                # Accumulate mean and std of training statistics
                keys = [key for key in acc_train_stats[model]]
                for key in keys:
                    acc_train_stats[model][f'mean_{key}'] = np.mean(acc_train_stats[model][key])
                    acc_train_stats[model][f'std_{key}'] = np.std(acc_train_stats[model][key])

                # Mean and std of training time per epoch
                acc_train_stats[model]["mean_epoch_time"] = np.mean(epoch_time)
                acc_train_stats[model]["std_epoch_time"] = np.std(epoch_time)

                # Keep only statistics
                acc_train_stats[model] = {key: value for key, value in acc_train_stats[model].items()
                                          if key.startswith("mean") or key.startswith("std")}

                # Write the dictionary to a JSON file
                with open(f'{folder_path}/acc_train_stats.json', "w") as file:
                    json.dump(acc_train_stats[model], file)


train_models_fixed()
train_models_learned()

