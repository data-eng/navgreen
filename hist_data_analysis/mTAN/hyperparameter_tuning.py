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
file_handler = logging.FileHandler('hist_data_analysis/mTAN/hyperparameter_tuning_logger_new_data.log')
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


def tuning(dim, task, X_cols, y_cols, pvt_cols):

    # Get all tunable parameters ...
    # architecture wise:
    num_heads_choices = [1, 2, 4, 8] #[2] #
    rec_hidden_choices = [8, 16, 32] # [16] #
    # training wise:
    batch_size_choices = [8, 16, 32] # [16] #
    learning_rate_choices = [0.0005, 0.001, 0.005, 0.01] # [0.001] #

    sequence_length = 24 // 3
    embed_time = sequence_length

    # Other parameters
    epochs = 30
    patience = 10
    validation_set_percentage = 0.2


    # Configure the different choices for the tunable parameters
    hyper_config = {
        "batch_size": batch_size_choices,
        "lr": learning_rate_choices,
        "num_heads": num_heads_choices,
        "rec_hidden": rec_hidden_choices
    }

    hyperparameters = dict()
    # Get all possible combination of parameters
    # parameter_combinations = it.product(*(hyper_config[param] for param in hyper_config))

    best_model_parameters = hyperparameters.copy()

    best_validation_loss = float('inf')


    df_path_train = "data/training_set_classif.csv"
    train_df, _ = load_df(df_path=df_path_train, pvt_cols=pvt_cols, parse_dates=["DATETIME"], normalize=True, y_cols=y_cols)

    # MSE loss
    criterion = nn.MSELoss()

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
    for counter, parameters in enumerate(list(parameter_combinations)):

        # make current parameters
        hyperparameters["batch_size"] = parameters[0]
        hyperparameters["lr"] = parameters[1]
        hyperparameters["num_heads"] = parameters[2]
        hyperparameters["rec_hidden"] = parameters[3]

        batch_size = hyperparameters["batch_size"]
        learning_rate = hyperparameters["lr"]
        num_heads = hyperparameters["num_heads"]
        rec_hidden = hyperparameters["rec_hidden"]

        # Try training with these parameters
        print("\n")
        logger.info(f'Test number {counter+1} | Start testing parameter-combination : {hyperparameters}')

        # Create dataloaders for training and validation sets
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Configure model
        model = MtanGruRegr(input_dim=dim, query=torch.linspace(0, 1., embed_time), nhidden=rec_hidden,
                            embed_time=embed_time, num_heads=num_heads, device=device, output_len=sequence_length).to(device)

        # Train the model
        try:
            training_loss, validation_loss = train(model=model, train_loader=train_loader, val_loader=val_loader,
                                                   checkpoint_pth=None, criterion=criterion, task=task,
                                                   learning_rate=learning_rate, epochs=epochs, patience=patience)
        except Exception as e:
            logger.info(e)
            training_loss, validation_loss = float('inf'), float('inf')

        logger.info(f'Test number {counter+1} | Training Loss : {training_loss:.6f} '
                    f'&  Validation Loss : {validation_loss:.6f}')

        # Save these parameters if they give us a smaller validation set-loss
        if validation_loss < best_validation_loss:
            logger.info('New best parameters found!\n')
            best_validation_loss = validation_loss
            best_model_parameters = hyperparameters.copy()

    logger.info(f'Parameters that cause the lowest validation error ({best_validation_loss:.6f}) are: {best_model_parameters}')

    # Write the dictionary to a JSON file
    with open(f'hist_data_analysis/mTAN/best_model_params_{task}.json', 'w') as file:
        json.dump(best_model_parameters, file)

    return best_model_parameters

def hyper_tuning():
    X_cols = ["humidity", "pressure", "feels_like", "temp", "wind_speed"]
    y_cols = ["Q_PVT"]

    pvt_cols = ["DATETIME"] + X_cols + y_cols

    dim = len(X_cols)
    best_model_parameters_pvt = tuning(dim=dim, task="pvt", X_cols=X_cols, y_cols=y_cols, pvt_cols=pvt_cols)

    logger.info(f"Best model parameters for pvt task are: {best_model_parameters_pvt}")
