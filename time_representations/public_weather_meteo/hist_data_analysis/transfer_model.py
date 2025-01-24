import yaml

from transformer.tune import main_loop as tune_transformer


def tune_models():
    with open("../config.yaml", "r") as config:
        config = yaml.safe_load(config)

    seeds = config["seeds"]
    bins = config["bins"]

    models = ["transformer"]
    model_tune = {"transformer": tune_transformer}

    for bin in bins:
        for seed in seeds:
            for model in models:
                print(f'Start tuning bin={bin} with model "{model}" for seed={seed}')
                model_tune[model](seed, [bin])
                print(f'End tuning bin={bin} with model "{model}" for seed={seed}')


tune_models()
