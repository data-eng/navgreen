import time
import json
import numpy as np

from hist_data_analysis.interpolation.train_and_test_classif import main_loop_train as train_interpolation
from hist_data_analysis.mTAN.train_and_test_classif import main_loop_train as train_mTAN
from hist_data_analysis.transformer.train import main_loop as train_transformer

# Epoch > 1

def train_models():
    #seeds = [6, 72, 157, 838, 1214, 1505]
    seeds = [6, 15]
    models = ["transformer", "interpolation", "mTAN"]
    #models = ["interpolation"]

    train_times = {"transformer" : dict(), "interpolation" : dict(), "mTAN": dict()}
    acc_train_stats = {"transformer": dict(), "interpolation": dict(), "mTAN": dict()}

    model_train = {"transformer" : train_transformer,
                   "interpolation" : train_interpolation,
                   "mTAN": train_mTAN}

    # Start training the models for each seed. The training information is stored within the folder 'models/{model_name}'
    for seed in seeds:
        for model in models:
            print(f'Start training model "{model}" for seed={seed}')
            start_time = time.time()
            model_train[model](seed)
            # Store training time for this model
            train_times[model][seed] = time.time() - start_time
            print(f'End training model "{model}" for seed={seed} [training time:{train_times[model][seed]:.2f}]')

    # Calculate mean and std of training statistics for each model
    for model in models:
        epoch_time = []
        folder_path = f'models/{model}'

        for s in seeds:
            seed = str(s)
            with open(f'{folder_path}/{seed}/train_checkpoints.json', "r") as file:
                checkpoint_data = json.load(file)
                # Training time per epoch
                epoch_time += [train_times[model][s] / checkpoint_data["epochs"]]

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



def eval_models():
    pass