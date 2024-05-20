import yaml

from mTAN.train_and_test_classif import main_loop_test_time_repr as test_time_repr


def eval_models():
    with open("../config.yaml", "r") as config:
        config = yaml.safe_load(config)

    bins = config["bins"]
    '''
    sin_best = ["sin", "1013"]
    time_representation = sin_best[0]
    seed = sin_best[1]

    for bin, _ in bins:
        print(f'Start test bin={bin} with model "mTAN" for seed={seed} and '
              f'time representation={time_representation}')
        test_time_repr(seed, [bin], time_representation)
        print(f'End test bin={bin} with model "mTAN" for seed={seed} and '
              f'time representation={time_representation}')
    '''
    pulse_best = ["tr_pulse", "1013"]
    time_representation = pulse_best[0]
    seed = pulse_best[1]

    for bin, _ in bins:
        print(f'Start test bin={bin} with model "mTAN" for seed={seed} and '
              f'time representation={time_representation}')
        test_time_repr(seed, [bin], time_representation)
        print(f'End test bin={bin} with model "mTAN" for seed={seed} and '
              f'time representation={time_representation}')


eval_models()


