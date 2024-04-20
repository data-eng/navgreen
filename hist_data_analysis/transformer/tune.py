import torch
import itertools as it
from torch.utils.data import DataLoader
import logging
from .model import Transformer
from .train import train
from .loader import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s:%(lineno)d:%(levelname)s:%(name)s:%(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f'Device is {device}')

def tune(data, static, config):
    ds_train, ds_val = data
    combs = list(it.product(*config.values()))
    _, epochs, patience, classes, weights, seed = static.values()
    num_classes = len(classes)
    
    best_val_loss = float('inf')
    best_params = None
    results = []

    for c, hyperparams in enumerate(combs):
        batch_size, lr, nhead, num_layers, dim_feedforward, dropout, optimizer = hyperparams

        if dim_feedforward % nhead != 0:
            logger.debug(f'Skipping combination {c+1} with incompatible parameters: {hyperparams}')
            continue

        logger.info(f'\nTuning case {c+1} with: {hyperparams}')
    
        dl_train = DataLoader(ds_train, batch_size, shuffle=True)
        dl_val = DataLoader(ds_val, batch_size, shuffle=False)

        model=Transformer(in_size=len(params["X"])+len(params["t"]), 
                          out_size=num_classes,
                          nhead=nhead, 
                          num_layers=num_layers, 
                          dim_feedforward=dim_feedforward, 
                          dropout=dropout)

        train_loss, val_loss = train(data=(dl_train, dl_val),
                                     classes=classes,
                                     epochs=epochs,
                                     patience=patience,
                                     lr=lr,
                                     criterion=utils.WeightedCrossEntropyLoss(weights),
                                     model=model,
                                     optimizer=optimizer,
                                     scheduler=("StepLR", 1.0, 0.98),
                                     seed=seed,
                                     visualize=False)
        
        results.append({'params': hyperparams, 'train_loss': train_loss, 'val_loss': val_loss})

        if val_loss < best_val_loss:
            logger.info('New best parameters found!\n')
            best_val_loss = val_loss

            best_params = best_params = {'batch_size': batch_size, 
                                         'lr': lr, 'nhead': nhead, 
                                         'num_layers': num_layers, 
                                         'dim_feedforward': dim_feedforward, 
                                         'dropout': dropout,
                                         'optimizer': optimizer}

        utils.save_json(data=results, filename='static/tuning_results.json')
        utils.save_json(data=best_params, filename='static/best_params.json')
        
    return best_params

def main():
    path = "data/owm+plc/training_set_classif.csv"

    df = load(path=path, parse_dates=["DATETIME"], normalize=True)
    df_prep = prepare(df, phase="train")

    static = {'seq_len': 1440 // 180, 
              'epochs': 15, 
              'patience': 50, 
              'classes': ["< 0.42 KWh", "< 1.05 KWh", "< 1.51 KWh", "< 2.14 KWh", ">= 2.14 KWh"], 
              'weights': utils.load_json(filename='static/weights.json'),
              'seed': 349}
    
    config = {'batch_size': [1, 8, 16], 
              'lr': [1e-3, 5e-4], 
              'nhead': [1,2,3,4,6,12],
              'num_layers': [1,2],
              'dim_feedforward': [256, 1024, 2048], 
              'dropout': [0, 0.1],
              'optimizer':["Adam", "AdamW"]}

    ds = TSDataset(df=df_prep, seq_len=static["seq_len"], X=params["X"], t=params["t"], y=params["y"])
    ds_train, ds_val = split(ds, vperc=0.2)

    best_params = tune(data=(ds_train, ds_val), static=static, config=config)
    logger.info(f"Best model parameters: {best_params}")