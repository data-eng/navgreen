import numpy as np
import json
import itertools as it
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from .train_and_test import train
from .model import MtanGruRegr
from .data_loader import load_df, TimeSeriesDataset

import logging

# Configure logger and set its level
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(lineno)d:%(message)s')
# Set log file, its level and format
file_handler = logging.FileHandler('./hyperparameter_tuning_logger.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
# Set stream its level and format
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set fixed seed for reproducibility
torch.manual_seed(1505)
np.random.seed(1505)
torch.cuda.manual_seed(1505)

hp_cols = ["DATETIME", "BTES_TANK", "DHW_BUFFER", "POWER_HP", "Q_CON_HEAT"]
pvt_cols = ["DATETIME", "OUTDOOR_TEMP", "PYRANOMETER", "DHW_BOTTOM", "POWER_PVT", "Q_PVT"]

def tuning(dim, task, X_cols, y_cols):

    # Get all tunable parameters ...
    # architecture wise:
    num_heads_choices = [2, 4, 8]
    rec_hidden_choices = [16, 32]
    embed_time_choices = [32, 64, 128]
    # training wise:
    sequence_length_choices = [10, 20]
    batch_size_choices = [16, 32, 64]
    learning_rate_choices = [0.001, 0.01]
    grp_choices = ["10min", "30min", "60min", "120min"]

    # Other parameters
    epochs = 5
    patience = 2
    validation_set_percentage = 0.2


    # Configure the different choices for the tunable parameters
    hyper_config = {
        "batch_size": batch_size_choices,
        "lr": learning_rate_choices,
        "num_heads": num_heads_choices,
        "rec_hidden": rec_hidden_choices,
        "embed_time": embed_time_choices
    }

    hyperparameters = dict()
    # Get all possible combination of parameters
    # parameter_combinations = it.product(*(hyper_config[param] for param in hyper_config))

    best_model_parameters = hyperparameters.copy()

    best_validation_loss = float('inf')

    counter = 0

    for grp in grp_choices:

         # Configure group in a different loop in order not to load the dataframe all the time
        df_path_train = "data/training_set_before_conv.csv"
        (df_train, df_hp_train, df_pvt_train), mean_stds = load_df(df_path=df_path_train, hp_cols=hp_cols,
                                                                   pvt_cols=pvt_cols, parse_dates=["Date&time"],
                                                                   normalize=True, grp=grp, hist_data=True)

        train_df = df_hp_train if task == "hp" else df_pvt_train

        for sequence_length in sequence_length_choices:
            # Create a dataset and dataloader
            training_dataset = TimeSeriesDataset(dataframe=train_df, sequence_length=sequence_length,
                                                 X_cols=X_cols, y_cols=y_cols)

            # Get the total number of samples and compute size of each corresponding set
            total_samples = len(training_dataset)
            validation_size = int(validation_set_percentage * total_samples)
            train_size = total_samples - validation_size

            # Use random_split to create training and validation datasets
            train_dataset, val_dataset = random_split(training_dataset, [train_size, validation_size])

            # Get all possible combination of parameters
            parameter_combinations = it.product(*(hyper_config[param] for param in hyper_config))

            # Iterate through the rest hyperparameter combinations
            for parameters in list(parameter_combinations):

                counter += 1
                # make current parameters
                hyperparameters["batch_size"] = parameters[0]
                hyperparameters["lr"] = parameters[1]
                hyperparameters["num_heads"] = parameters[2]
                hyperparameters["rec_hidden"] = parameters[3]
                hyperparameters["embed_time"] = parameters[4]
                hyperparameters["sequence_length"] = sequence_length
                hyperparameters["grp"] = grp

                batch_size = hyperparameters["batch_size"]
                learning_rate = hyperparameters["lr"]
                num_heads = hyperparameters["num_heads"]
                rec_hidden = hyperparameters["rec_hidden"]
                embed_time = hyperparameters["embed_time"]

                # Try training with these parameters
                print("\n")
                logger.info(f'Test number {counter} | Start testing parameter-combination : {hyperparameters}')

                # Create dataloaders for training and validation sets
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

                # Configure model
                model = MtanGruRegr(input_dim=dim, query=torch.linspace(0, 1., embed_time), nhidden=rec_hidden,
                                    embed_time=embed_time, num_heads=num_heads, device=device).to(device)
                # MSE loss
                criterion = nn.MSELoss()
                # Train the model
                try:
                    training_loss, validation_loss = train(model=model, train_loader=train_loader, val_loader=val_loader,
                        	                               checkpoint_pth=None, criterion=criterion, task=task,
                                	                       learning_rate=learning_rate, epochs=epochs, patience=patience)
                except Exception as e:
                    logger.info(e)
                    training_loss, validation_loss = float('inf'), float('inf')

                logger.info(f'Test number {counter} | Training Loss : {training_loss:.6f} '
                            f'&  Validation Loss : {validation_loss:.6f}\n')

                # Save these parameters if they give us a smaller validation set-loss
                if validation_loss < best_validation_loss:
                    logger.info('New best parameters found!\n')
                    best_validation_loss = validation_loss
                    best_model_parameters = hyperparameters.copy()

    logger.info(f'Parameters that cause the lowest validation error ({best_validation_loss}) are: {best_model_parameters}')

    # Write the dictionary to a JSON file
    with open(f'./best_model_params_{task}.json', 'w') as file:
        json.dump(best_model_parameters, file)

    return best_model_parameters

def hyper_tuning():
    logger.info(f"Start tuning for Task 1 (hp)")
    best_model_parameter_hp = tuning(dim=2, task="hp", X_cols=["BTES_TANK", "DHW_BUFFER"],
                               y_cols=["POWER_HP", "Q_CON_HEAT"])
    logger.info(f"Best model parameters for hp task are: {best_model_parameter_hp}")

    logger.info(f"Start tuning for Task 2 (pvt)")
    best_model_parameters_pvt = tuning(dim=3, task="pvt",
                                X_cols=["OUTDOOR_TEMP", "PYRANOMETER", "DHW_BOTTOM"], y_cols=["POWER_PVT", "Q_PVT"])
    logger.info(f"Best model parameters for pvt task are: {best_model_parameters_pvt}")