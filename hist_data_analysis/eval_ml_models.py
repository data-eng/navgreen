import json
import numpy as np
import yaml

from hist_data_analysis.interpolation.train_and_test_classif import main_loop_test as test_interpolation
from hist_data_analysis.mTAN.train_and_test_classif import main_loop_test as test_mTAN
from hist_data_analysis.transformer.test import main_loop as test_transformer


def eval_models():
    # seeds = [6, 72, 157, 838, 1214, 1505]

    with open("hist_data_analysis/config.yaml", "r") as config:
        config = yaml.safe_load(config)

    seeds = config["seeds"]
    models = ["transformer", "interpolation", "mTAN"]

    acc_test_stats = {"transformer": dict(), "interpolation": dict(), "mTAN": dict()}

    model_test = {"transformer": test_transformer,
                   "interpolation": test_interpolation,
                   "mTAN": test_mTAN}

    # Start testing the models for each seed. The testing information is stored within the folder 'models/{model_name}'
    for seed in seeds:
        for model in models:
            print(f'Start testing model "{model}" for seed={seed}')
            model_test[model](seed)
            print(f'End testing model "{model}" for seed={seed}')

    # Calculate mean and std of testing statistics for each model
    for model in models:
        folder_path = f'models/{model}'

        for s in seeds:
            seed = str(s)
            with open(f'{folder_path}/{seed}/test_checkpoints.json', "r") as file:
                checkpoint_data = json.load(file)

                for key in checkpoint_data:
                    if key not in ["seed"]:
                        try:
                            acc_test_stats[model][key] += [checkpoint_data[key]]
                        except KeyError:
                            acc_test_stats[model][key] = [checkpoint_data[key]]

        # Accumulate mean and std of testing statistics
        keys = [key for key in acc_test_stats[model]]
        for key in keys:
            acc_test_stats[model][f'mean_{key}'] = np.mean(acc_test_stats[model][key])
            acc_test_stats[model][f'std_{key}'] = np.std(acc_test_stats[model][key])

        # Keep only statistics
        acc_test_stats[model] = {key: value for key, value in acc_test_stats[model].items()
                                  if key.startswith("mean") or key.startswith("std")}

        # Write the dictionary to a JSON file
        with open(f'{folder_path}/acc_test_stats.json', "w") as file:
            json.dump(acc_test_stats[model], file)