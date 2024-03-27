import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from random import SystemRandom
from .model import MtanGruRegr

from .data_loader import load_df, TimeSeriesDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device is {device}')


torch.manual_seed(1505)
np.random.seed(1505)
torch.cuda.manual_seed(1505)

hp_cols = ["DATETIME", "BTES_TANK", "DHW_BUFFER", "POWER_HP", "Q_CON_HEAT"]
pvt_cols = ["DATETIME", "OUTDOOR_TEMP", "PYRANOMETER", "DHW_BOTTOM", "POWER_PVT", "Q_PVT"]

def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    for (X, masks), y in dataloader:
        X, y = X.to(device), y.to(device)
        observed_tp = X[:, :, -1]
        with torch.no_grad():
            out = model(X, observed_tp, masks)
            y_ = y[:, -1, :]
            total_loss += criterion(out, y_)

    return total_loss / len(dataloader)


def train(model, train_loader, val_loader, checkpoint_pth, experiment_id, criterion, learning_rate = 0.01, epochs = 5, patience=2):

    # Early stopping variables
    best_val_loss = float('inf')
    final_train_loss = float('inf')
    epochs_without_improvement = 0

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if checkpoint_pth is not None:
        checkpoint = torch.load(checkpoint_pth)
        model.load_state_dict(checkpoint['rec_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('loading saved weights', checkpoint['epoch'])

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0

        start_time = time.time()
        for (X, masks), y in train_loader:
            X, y = X.to(device), y.to(device)
            observed_tp  = X[:, :, -1]

            out = model(X, observed_tp, masks)
            y_ = y[:, -1, :]
            loss = criterion(out, y_)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Get validation loss
        val_loss = evaluate(model, val_loader, criterion)
        # Compute average training loss
        average_loss = total_loss / len(train_loader)
        print(f'Epoch: {epoch} | Training Loss: {average_loss:.4f}, Validation Loss: {val_loss:.4f}, '
              f'time : {(time.time() - start_time)/60:.2f} minutes')

        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            final_train_loss = average_loss
            epochs_without_improvement = 0

            torch.save({
                'epoch': epoch,
                'rec_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': average_loss,
            }, f'epoch_{epoch}_{experiment_id}.pth')
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping after {epoch + 1} epochs without improvement. Patience is {patience}.")
            break

    print("Training complete!")

    return final_train_loss, best_val_loss



def main_loop():

    experiment_id = int(SystemRandom().random() * 100000)
    # Parameters:
    freq = 10.0
    num_heads = 1
    rec_hidden = 32
    embed_time = 128
    learn_emb = True
    dim = 2

    sequence_length = 10
    batch_size = 32
    validation_set_percentage = 0.2

    grp = "5T"

    df_path = "data/DATA_FROM_PLC.csv"
    (df, df_hp, df_pvt), mean_stds = load_df(df_path=df_path, hp_cols=hp_cols, pvt_cols=pvt_cols,
                                             parse_dates=["Date&time"], normalize=True, grp=grp, hist_data=True)
    df_hp.to_csv("hp_df.csv")

    train_df = pd.read_csv("hp_df.csv", parse_dates=['DATETIME'], index_col='DATETIME')

    X_hp_cols = ["BTES_TANK", "DHW_BUFFER"]
    y_hp_cols = ["POWER_HP", "Q_CON_HEAT"]

    # X_pvt_cols = ["OUTDOOR_TEMP", "PYRANOMETER", "DHW_BOTTOM"]
    # y_pvt_cols = ["POWER_PVT", "Q_PVT"]

    # Create a dataset and dataloader
    training_dataset = TimeSeriesDataset(dataframe=train_df, sequence_length=sequence_length,
                                         X_cols=X_hp_cols, y_cols=y_hp_cols)

    # Get the total number of samples and compute size of each corresponding set
    total_samples = len(training_dataset)
    validation_size = int(validation_set_percentage * total_samples)
    train_size = total_samples - validation_size

    # Use random_split to create training and validation datasets
    train_dataset, val_dataset = random_split(training_dataset, [train_size, validation_size])

    # Create dataloaders for training and validation sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print("Train dataloader loaded")
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print("Validation dataloader loaded")

    # Configure model
    model = MtanGruRegr(input_dim=dim, query=torch.linspace(0, 1., 128), nhidden=rec_hidden, embed_time=embed_time,
                        num_heads=num_heads, learn_emb=learn_emb, freq=freq, device=device).to(device)
    # MSE loss
    criterion = nn.MSELoss()
    # Train the model
    training_loss, validation_loss =  train(model=model, train_loader=train_loader, val_loader=val_loader,
                                            checkpoint_pth=None, experiment_id=experiment_id, criterion=criterion)

    print(f'Final Training Loss : {training_loss:.6f} &  Validation Loss : {validation_loss:.6f}\n')
