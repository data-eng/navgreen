import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim

from .data_loader import load_df, TimeSeriesDataset
from .model import LSTMRegressor

import time
import logging

# Configure logger and set its level
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# Configure format
formatter = logging.Formatter('%(asctime)s:%(lineno)d:%(levelname)s:%(name)s:%(message)s')
# Configure stream handler
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
# Add handler to logger
logger.addHandler(stream_handler)

# Set a fixed seed for reproducibility
torch.manual_seed(15)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f'Device is {device}')

hp_cols = ["DATETIME", "BTES_TANK", "DHW_BUFFER", "POWER_HP", "Q_CON_HEAT"]
pvt_cols = ["DATETIME", "OUTDOOR_TEMP", "PYRANOMETER", "DHW_BOTTOM", "POWER_PVT", "Q_PVT"]


# POWER_HP= f (BTES_TANK, DHW_BUFFER, WATER_IN_EVAP, WATER_IN_COND, OUTDOOR_TEMP, COMPRESSOR_HZ)
# Q_CON_HEAT=( WATER_OUT_COND, WATER_IN_COND, FLOW_CONDENSER, DHW_BUFFER, POWER_HP)

# POWER_PVT= f (OUTDOOR_TEMP, PYRANOMETER, PVT_IN, PVT_OUT)
# Q_PVT= f(OUTDOOR_TEMP, PYRANOMETER, PVT_IN, PVT_OUT, FLOW_PVT)



def train(model, dataloader_train, dataloader_valid, learning_rate, epochs, patience):
    model = model.to(device)

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Early stopping variables
    best_val_loss = float('inf')
    final_train_loss = float('inf')
    epochs_without_improvement = 0

    # Training loop
    for epoch in range(epochs):

        model.train()
        total_loss = 0.0
        start_time = time.time()

        for X, y in dataloader_train:
            # Forward pass
            X, y = X.to(device), y.to(device)
            predictions = model(X)

            # Compute the loss
            y_ = y[:, -1, :]  # Take the output from the last time step
            loss = criterion(predictions, y_)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Print the average loss for the epoch
        average_loss = total_loss / len(dataloader_train)
        logger.info(f"Epoch [{epoch + 1}/{epochs}], Training Loss: {average_loss:.6f}, "
                    f"Time: {(time.time() - start_time) / 60:.2f} minutes")

        # Validation
        val_loss = evaluate(model, dataloader_valid, criterion)
        logger.info(f"Epoch [{epoch + 1}/{epochs}], Validation Loss: {val_loss:.6f}")

        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            final_train_loss = average_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            logger.info(f"Early stopping after {epoch + 1} epochs without improvement. Patience is {patience}.")
            break

    logger.info("Training complete!")

    return final_train_loss, best_val_loss


def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        # Forward pass
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            # Forward pass
            predictions = model(X)
            # Compute the loss
            y_ = y[:, -1, :]  # Take the output from the last time step
            loss = criterion(predictions, y_)

            # Accumulate loss
            total_loss += loss.item()

    return total_loss / len(dataloader)



def main_loop():
    grp = "2T" # 2T is 2 minutes

    '''
    df_path = "data/DATA_FROM_PLC.csv"
    (df, df_hp, df_pvt), _ = load_df(df_path, hp_cols, pvt_cols, normalize=True, grp=grp)  

    df_hp.to_csv("hp_df.csv")

    print(df.shape, df.columns)
    print(df_hp.shape, df_hp.columns)
    print(df_pvt.shape, df_pvt.columns)
    '''
    train_df = pd.read_csv("hp_df.csv", parse_dates=['DATETIME'], index_col='DATETIME')

    # Specify the sequence length
    sequence_length = 5

    X_hp_cols = ["BTES_TANK", "DHW_BUFFER"]
    y_hp_cols = ["POWER_HP", "Q_CON_HEAT"]

    # X_pvt_cols = ["OUTDOOR_TEMP", "PYRANOMETER", "DHW_BOTTOM"]
    # y_pvt_cols = ["POWER_PVT", "Q_PVT"]

    # Hyperparameters
    # Not tunable
    epochs = 1
    patience = 2
    # Tunable: Model related
    input_size = 10
    hidden_size = 64
    num_layers = 1
    output_size = 2
    # Tunable: Training related
    sequence_length = 5
    learning_rate = 0.001
    batch_size = 4
    validation_set_percentage = 0.2

    # Create a dataset and dataloader
    training_dataset = TimeSeriesDataset(dataframe=train_df, sequence_length=sequence_length,
                                         X_cols=X_hp_cols, Y_cols=y_hp_cols)

    # Get the total number of samples and compute size of each corresponding set
    total_samples = len(training_dataset)
    validation_size = int(validation_set_percentage * total_samples)
    train_size = total_samples - validation_size

    # Use random_split to create training and validation datasets
    train_dataset, val_dataset = random_split(training_dataset, [train_size, validation_size])

    # Create dataloaders for training and validation sets
    dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    logger.info("Train dataloader loaded")
    dataloader_val = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    logger.info("Validation dataloader loaded")

    # Create an instance of the LSTMRegressor
    model = LSTMRegressor(input_size, hidden_size, num_layers, output_size)

    # Train the model
    training_loss, validation_loss = train(model=model, dataloader_train=dataloader_train, dataloader_valid=dataloader_val,
                                           patience=patience, learning_rate=learning_rate, epochs=epochs)

    logger.info(f'Final Training Loss : {training_loss:.6f} &  Validation Loss : {validation_loss:.6f}\n')

    logger.info("Training complete!")
