import json
import numpy as np
import yaml

from transformer.test import main_loop as test_transformer


def eval_models():
    with open("../config.yaml", "r") as config:
        config = yaml.safe_load(config)

    seeds = config["seeds"]
    bins = config["bins"]

    models = ["transformer"]
    model_test = {"transformer": test_transformer}

    # Start testing the models for each seed and binning.
    # The testing information is stored within the folder 'models/{bin}/{model_name}'
    for bin in bins:
        for seed in seeds:
            for model in models:
                print(f'Start test bin={bin} with model "{model}" for seed={seed}')
                model_test[model](seed, [bin])
                print(f'End test bin={bin} with model "{model}" for seed={seed}')

    # Calculate mean and std of testin statistics for each model
    for bin in bins:
        bin_folder_path = f'models/{bin}'
        acc_test_stats = {"transformer": dict()}
        for model in models:
            epoch_time = []
            folder_path = f'{bin_folder_path}/{model}'

            for s in seeds:
                seed = str(s)
                with open(f'{folder_path}/{seed}/test_checkpoints.json', "r") as file:
                    checkpoint_data = json.load(file)

                    for key in checkpoint_data:
                        if key not in ["seed"]:
                            try: acc_test_stats[model][key] += [checkpoint_data[key]]
                            except KeyError: acc_test_stats[model][key] = [checkpoint_data[key]]

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

eval_models()
