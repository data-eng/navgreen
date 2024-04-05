import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from .model import MtanGruRegr
from .data_loader import load_df, TimeSeriesDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device is {device}')


torch.manual_seed(1505)
np.random.seed(1505)
torch.cuda.manual_seed(1505)

hp_cols = ["DATETIME", "BTES_TANK", "DHW_BUFFER", "POWER_HP", "Q_CON_HEAT"]
pvt_cols = ["DATETIME", "OUTDOOR_TEMP", "PYRANOMETER", "DHW_BOTTOM", "POWER_PVT", "Q_PVT"]


def evaluate(model, dataloader, criterion, plot=False, pred_values=None):
    model.eval()
    total_loss = 0
    true_values = []
    predicted_values = []
    for (X, masks_X, observed_tp), y in dataloader:
        X, masks_X, y = X.to(device), masks_X.to(device), y.to(device)

        with torch.no_grad():
            out = model(X, observed_tp, masks_X)
            y_ = y[:, -1, :]
            total_loss += criterion(out, y_)
            true_values.append(y_.cpu().numpy())
            predicted_values.append(out.cpu().numpy())

    true_values = np.concatenate(true_values, axis=0)
    predicted_values = np.concatenate(predicted_values, axis=0)

    if plot:
        assert pred_values is not None

        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.scatter(true_values[:, 0], predicted_values[:, 0], color='lightblue')
        plt.xlabel("True Value")
        plt.ylabel("Predicted Value")
        plt.title(pred_values[0])

        plt.subplot(1, 2, 2)
        plt.scatter(true_values[:, 1], predicted_values[:, 1], color='lightcoral')
        plt.xlabel("True Value")
        plt.ylabel("Predicted Value")
        plt.title(pred_values[1])

        plt.tight_layout()
        plt.show()

    return total_loss / len(dataloader)


def train(model, train_loader, val_loader, checkpoint_pth, criterion, task, learning_rate = 0.01, epochs = 5, patience=3):

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
        for (X, masks_X, observed_tp), y in train_loader:
            X, masks_X, y = X.to(device), masks_X.to(device), y.to(device)
            out = model(X, observed_tp, masks_X)
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
        print(f'Epoch {epoch} | Training Loss: {average_loss:.6f}, Validation Loss: {val_loss:.6f}, '
              f'Time : {(time.time() - start_time)/60:.2f} minutes')

        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            final_train_loss = average_loss
            epochs_without_improvement = 0

            torch.save({
                'epoch': epoch,
                'mod_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': average_loss,
            }, f'best_model_{task}.pth')
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping after {epoch} epochs without improvement. Patience is {patience}.")
            break

    print("Training complete!")

    return final_train_loss, best_val_loss



def main_loop():

    validation_set_percentage = 0.2

    print("TASK 1 | Train and evaluate on HP related prediction")
    task = "hp"

    with open("./best_model_params_hp.json", 'r') as file:
        params = json.load(file)

    dim = 2
    # Parameters:
    num_heads = params["num_heads"]
    rec_hidden = params["rec_hidden"]
    embed_time = params["embed_time"]

    sequence_length = params["sequence_length"]
    batch_size = params["batch_size"]
    grp = params["grp"]

    lr = params["lr"]
    epochs = 30
    patience = 8

    print(f"Parameters : num_heads = {num_heads}, rec_hidden = {rec_hidden}, embed_time = {embed_time}, "
          f"sequence_length = {sequence_length}, batch_size = {batch_size}, grp = {grp}")

    df_path_train = "data/training_set_before_conv.csv"
    (_, df_hp_train, _), mean_stds = load_df(df_path=df_path_train, hp_cols=hp_cols, pvt_cols=pvt_cols,
                                             parse_dates=["Date&time"], normalize=True, grp=grp, hist_data=True)

    df_path_test = "data/test_set_before_conv.csv"
    (_, df_hp_test, _), _ = load_df(df_path=df_path_test, hp_cols=hp_cols, pvt_cols=pvt_cols, parse_dates=["Date&time"],
                                    normalize=True, grp=grp, hist_data=True, stats=mean_stds)

    df_hp_train.to_csv("hp_df_train.csv")
    df_hp_test.to_csv("hp_df_test.csv")

    train_df = pd.read_csv("hp_df_train.csv", parse_dates=['DATETIME'], index_col='DATETIME')
    test_df = pd.read_csv("hp_df_test.csv", parse_dates=['DATETIME'], index_col='DATETIME')

    X_cols = ["BTES_TANK", "DHW_BUFFER"]
    y_cols = ["POWER_HP", "Q_CON_HEAT"]

    # Create a dataset and dataloader
    training_dataset = TimeSeriesDataset(dataframe=train_df, sequence_length=sequence_length,
                                         X_cols=X_cols, y_cols=y_cols, final_train=True)

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
    model = MtanGruRegr(input_dim=dim, query=torch.linspace(0, 1., embed_time), nhidden=rec_hidden, embed_time=embed_time,
                        num_heads=num_heads, device=device).to(device)
    # MSE loss
    criterion = nn.MSELoss()
    # Train the model
    training_loss, validation_loss =  train(model=model, train_loader=train_loader, val_loader=val_loader,
                                            checkpoint_pth=None, criterion=criterion, task=task, learning_rate=lr,
                                            epochs=epochs, patience=patience)

    print(f'Final Training Loss : {training_loss:.6f} &  Validation Loss : {validation_loss:.6f}\n')

    # Create a dataset and dataloader
    testing_dataset = TimeSeriesDataset(dataframe=test_df, sequence_length=sequence_length,
                                         X_cols=y_cols, y_cols=y_cols)

    test_loader = DataLoader(testing_dataset, batch_size=batch_size, shuffle=False)
    print("Test dataloader loaded")

    trained_model = MtanGruRegr(input_dim=dim, query=torch.linspace(0, 1., embed_time), nhidden=rec_hidden,
                                embed_time=embed_time, num_heads=num_heads, device=device).to(device)
    checkpoint = torch.load(f'best_model_{task}.pth')
    trained_model.load_state_dict(checkpoint['mod_state_dict'])

    # Test model's performance on unseen data
    testing_loss = evaluate(trained_model, test_loader, criterion, plot=True, pred_values=y_cols)

    print(f'Testing Loss (MSE) : {testing_loss:.6f}')


    print("\nTASK 2 | Train and evaluate on PVT related prediction")
    task = "pvt"

    with open("./best_model_params_pvt.json", 'r') as file:
        params = json.load(file)

    dim = 3
    # Parameters:
    num_heads = params["num_heads"]
    rec_hidden = params["rec_hidden"]
    embed_time = params["embed_time"]
    sequence_length = params["sequence_length"]
    batch_size = params["batch_size"]
    lr = params["lr"]
    grp = params["grp"]

    print(f"Parameters : num_heads = {num_heads}, rec_hidden = {rec_hidden}, embed_time = {embed_time}, "
          f"sequence_length = {sequence_length}, batch_size = {batch_size}, grp = {grp}")

    df_path_train = "data/training_set_before_conv.csv"
    (_, _, df_pvt_train), mean_stds = load_df(df_path=df_path_train, hp_cols=hp_cols, pvt_cols=pvt_cols,
                                              parse_dates=["Date&time"], normalize=True, grp=grp, hist_data=True)

    df_path_test = "data/test_set_before_conv.csv"
    (_, _, df_pvt_test), _ = load_df(df_path=df_path_test, hp_cols=hp_cols, pvt_cols=pvt_cols, parse_dates=["Date&time"],
                                     normalize=True, grp=grp, hist_data=True, stats=mean_stds)

    df_pvt_train.to_csv("pvt_df_train.csv")
    df_pvt_test.to_csv("pvt_df_test.csv")

    train_df = pd.read_csv("pvt_df_train.csv", parse_dates=['DATETIME'], index_col='DATETIME')
    test_df = pd.read_csv("pvt_df_test.csv", parse_dates=['DATETIME'], index_col='DATETIME')

    X_cols = ["OUTDOOR_TEMP", "PYRANOMETER", "DHW_BOTTOM"]
    y_cols = ["POWER_PVT", "Q_PVT"]

    # Create a dataset and dataloader
    training_dataset = TimeSeriesDataset(dataframe=train_df, sequence_length=sequence_length,
                                         X_cols=X_cols, y_cols=y_cols, final_train=True)

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
    model = MtanGruRegr(input_dim=dim, query=torch.linspace(0, 1., embed_time), nhidden=rec_hidden,
                        embed_time=embed_time, num_heads=num_heads, device=device).to(device)
    # MSE loss
    criterion = nn.MSELoss()
    # Train the model
    training_loss, validation_loss = train(model=model, train_loader=train_loader, val_loader=val_loader,
                                           checkpoint_pth=None, criterion=criterion, task=task, learning_rate=lr,
                                           epochs=epochs, patience=patience)

    print(f'Final Training Loss : {training_loss:.6f} &  Validation Loss : {validation_loss:.6f}\n')

    # Create a dataset and dataloader
    testing_dataset = TimeSeriesDataset(dataframe=test_df, sequence_length=sequence_length,
                                        X_cols=X_cols, y_cols=y_cols)

    test_loader = DataLoader(testing_dataset, batch_size=batch_size, shuffle=False)
    print("Test dataloader loaded")

    trained_model = MtanGruRegr(input_dim=dim, query=torch.linspace(0, 1., embed_time), nhidden=rec_hidden,
                        embed_time=embed_time, num_heads=num_heads, device=device).to(device)
    checkpoint = torch.load(f'best_model_{task}.pth')
    trained_model.load_state_dict(checkpoint['mod_state_dict'])

    # Test model's performance on unseen data
    testing_loss = evaluate(trained_model, test_loader, criterion, plot=True, pred_values=y_cols)

    print(f'Testing Loss (MSE) : {testing_loss:.6f}')
