import time

from hist_data_analysis.interpolation.train_and_test_classif import main_loop as train_interpolation
from hist_data_analysis.mTAN.train_and_test_classif import main_loop as train_mTAN
from hist_data_analysis.transformer.train import main_loop as train_transformer

# Epoch > 1
# Mean epoch per model and std
# Mean time per model and std
# Mean training loss and std
# Mean training metrics and std
# Mean testing metrics and std

def train_models():
    #seeds = [6, 72, 157, 838, 1214, 1505]
    seeds = [6, 16]

    train_times = {"transformer" : dict(),
                   "interpolation" : dict(),
                   "mTAN": dict()}
    model_train = {"transformer" : train_transformer,
                   "interpolation" : train_interpolation,
                   "mTAN": train_mTAN}

    for seed in seeds:
        for model in ["transformer", "interpolation", "mTAN"]:
            print(f'Start training model {model} for seed {seed}')
            start_time = time.time()
            model_train[model](seed)
            # Store training time for this model
            train_times[model][seed] = time.time() - start_time
            print(f'End training model {model} for seed {seed}. Training time {train_times[model][seed]}')


def eval_models():
    pass