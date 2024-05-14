import time
import json
import numpy as np
import yaml

"""
# YAML
time_reprs = {
        1: {"datetime": ["month", "hour"],
              "cors": ["sine", "cosine"],
              "uniqs": ["cosine", "cosine"],
              "args": [[12, 12, 1], [12, 12, 1]] 
              },

        2: {"datetime": ["date"],
              "cors": ["triangular_pulse"],
              "uniqs": ["linear"],
              "args": [[12, 12, 1]] 
              }
]
"""

from transformer.train import main_loop as train_transformer

def train_models():
    with open("../config.yaml", "r") as config:
        config = yaml.safe_load(config)

    time_reprs = config["time_repr"]
    seeds = config["seeds"]
    bins = config["bins"]

    models = ["transformer"]

    train_times = {}
    for time_repr_key, _ in time_reprs.items():
        train_times[time_repr_key] = {}
        for bin in bins:
            train_times[time_repr_key][bin] = {"transformer": {}}

    model_train = {"transformer": train_transformer}
    
    # Start training the models for each time_repr, seed and binning.
    # The training information is stored within the folder 'models/{time_repr}/{bin}/{model_name}'
    for time_repr_key, time_repr_value in time_reprs.items():
        for bin in bins:
            for seed in seeds:
                for model in models:
                    print(f'Start training time_repr={time_repr_key}, bin={bin} with model "{model}" for seed={seed}')
                    start_time = time.time()

                    time_repr = (time_repr_value['cors'], time_repr_value['uniqs'], [tuple(arg) for arg in time_repr_value['args']])

                    model_train[model](time_repr, seed, [bin])

                    # Store training time for this model
                    train_times[time_repr][bin][model][seed] = time.time() - start_time
                    print(f'End training time_repr={time_repr_key}, bin={bin} with model "{model}" for seed={seed} [training time:{train_times[time_repr_key][bin][model][seed]:.2f}]')

    # Calculate mean and std of training statistics for each model
    for time_repr_key, time_repr_value in time_reprs.items():
        for bin in bins:
            bin_folder_path = f'models/{time_repr_key}/{bin}'
            acc_train_stats = {"transformer": dict()}
            for model in models:
                epoch_time = []
                folder_path = f'{bin_folder_path}/{model}'

            for s in seeds:
                seed = str(s)
                with open(f'{folder_path}/{seed}/train_checkpoints.json', "r") as file:
                    checkpoint_data = json.load(file)
                    # Training time per epoch
                    epoch_time += [train_times[time_repr_key][bin][model][s] / checkpoint_data["epochs"]]

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

train_models()