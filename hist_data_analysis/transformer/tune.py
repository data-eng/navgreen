import json
import torch
import torch.nn as nn
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
    _, num_classes, epochs, patience = static.values()
    
    best_val_loss = float('inf')
    best_params = None
    results = []

    for c, hyperparams in enumerate(combs):
        logger.info(f'\nTuning case {c+1} with: {hyperparams}')

        batch_size, lr, nhead, num_layers, dim_feedforward, dropout = hyperparams

        # poetry run tune_transformer
    
        dl_train = DataLoader(ds_train, batch_size, shuffle=True)
        dl_val = DataLoader(ds_val, batch_size, shuffle=False)

        model=Transformer(in_size=len(params["X"])+len(params["t"]), 
                          out_size=num_classes,
                          nhead=nhead, 
                          num_layers=num_layers, 
                          dim_feedforward=dim_feedforward, 
                          dropout=dropout
                          )

        train_loss, val_loss = train(data=(dl_train, dl_val),
                                    num_classes=num_classes,
                                    epochs=epochs,
                                    patience=patience,
                                    lr=lr,
                                    criterion=nn.CrossEntropyLoss(),
                                    model=model,
                                    optimizer="AdamW",
                                    scheduler=("StepLR", 1.0, 0.98),
                                    path="models/transformer.pth",
                                    plot=False
                                    )
        
        results.append({'params': hyperparams, 'train_loss': train_loss, 'val_loss': val_loss})

        if val_loss < best_val_loss:
            logger.info('New best parameters found!\n')
            best_val_loss = val_loss
            best_params = best_params = {'batch_size': batch_size, 'lr': lr, 'nhead': nhead, 'num_layers': num_layers, 'dim_feedforward': dim_feedforward, 'dropout': dropout}
        
        return results, best_params

def main():
    path = "data/owm+plc/training_set_classif.csv"

    static = {'seq_len': 1440 // 180, 'num_classes': 5, 'epochs': 10, 'patience': 50}
    config = {'batch_size': [8, 16, 32], 'lr': [1e-3, 1e-4], 'nhead': [1,2,3,4], 'num_layers': [1,2,3,4], 'dim_feedforward': [8, 64, 256, 1024], 'dropout': [0.1]}

    df = load(path=path, parse_dates=["DATETIME"], normalize=True)
    df_prep = prepare(df, phase="train")

    ds = TSDataset(df=df_prep, seq_len=static["seq_len"], X=params["X"], t=params["t"], y=params["y"])
    ds_train, ds_val = split(ds, vperc=0.2)

    results, best_params = tune(data=(ds_train, ds_val),
                                static=static,
                                config=config
                                )
    
    with open('tuning_results.json', 'w') as f:
        json.dump(results, f)
    
    with open('best_params.json', 'w') as f:
        json.dump(best_params, f)
    
    logger.info(f"Best model parameters: {best_params}")