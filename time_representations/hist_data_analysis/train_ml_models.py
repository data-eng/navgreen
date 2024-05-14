import time
import json
import numpy as np
import yaml

# from interpolation.train_and_test_classif import main_loop_train as train_interpolation
from mTAN.train_and_test_classif import main_loop_train as train_mTAN
# from transformer.train import main_loop as train_transformer


def train_models():
    with open("../config.yaml", "r") as config:
        config = yaml.safe_load(config)

    seeds = [1]  # config["seeds"]
    bins = config["bins"]

    models = ["mTAN"]  # ["interpolation", "mTAN", "transformer"]

    train_times = {}
    for bin, _ in bins:
        # train_times[bin] = {"transformer" : dict(), "interpolation" : dict(), "mTAN": dict()}
        train_times[bin] = {"mTAN": dict()}

    model_train = {# "transformer" : train_transformer,
                   # "interpolation" : train_interpolation,
                   "mTAN": train_mTAN}

    # Start training the models for each seed and binning.
    # The training information is stored within the folder 'models/{bin}/{model_name}'
    for bin, weights in bins:
        for seed in seeds:
            for model in models:
                print(f'Start training bin={bin} with model "{model}" for seed={seed}')
                start_time = time.time()
                if model == "transformer": model_train[model](seed, [bin])
                else: model_train[model](seed, [bin], weights)
                # Store training time for this model
                train_times[bin][model][seed] = time.time() - start_time
                print(f'End training bin={bin} with model "{model}" for seed={seed} [training time:{train_times[bin][model][seed]:.2f}]')

    # Calculate mean and std of training statistics for each model
    for bin, _ in bins:
        bin_folder_path = f'models/{bin}'
        acc_train_stats = {"transformer": dict(), "interpolation": dict(), "mTAN": dict()}
        for model in models:
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


train_models()

