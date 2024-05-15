import yaml

from mTAN.train_and_test_classif import main_loop_test_time_repr as test_time_repr


def eval_models():
    with open("../config.yaml", "r") as config:
        config = yaml.safe_load(config)

    seeds = config["seeds"]
    bins = config["bins"]

    for bin, _ in bins:
        for seed in seeds:
            print(f'Start test bin={bin} with "mTAN" for seed={seed}')
            test_time_repr(seed, [bin])
            print(f'End test bin={bin} with "mTAN" for seed={seed}')

    for bin, _ in bins:
        bin_folder_path = f'models/{bin}'

        folder_path = f'{bin_folder_path}/mTAN'

        for s in seeds:
            seed = str(s)
            p = f'{folder_path}/{seed}'


eval_models()
