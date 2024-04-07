import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from .model import MtanGruRegr
from .data_loader import load_df, TimeSeriesDataset
from .utils import MaskedMSELoss, MaskedSmoothL1Loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, dataloader, criterion, plot=False, pred_value=None, characteristics=None, limits=None, name=None, params=None):
    model.eval()
    total_loss = 0
    true_values = []
    predicted_values = []
    masked_values = []

    for (X, masks_X, observed_tp), (y, mask_y) in dataloader:
        X, masks_X, y, mask_y = X.to(device), masks_X.to(device), y.to(device), mask_y.to(device)

        with torch.no_grad():
            out = model(X, observed_tp, masks_X)
            total_loss += criterion(out, y, mask_y)

            if out.shape[0] != X.shape[0]:
                out = out.unsqueeze(0)

            # Append masked true and predicted values
            true_values.append(y.cpu().numpy())
            predicted_values.append(out.cpu().numpy())
            masked_values.append(mask_y.cpu().numpy())

    true_values = np.concatenate(true_values, axis=0)
    predicted_values = np.concatenate(predicted_values, axis=0)
    masked_values = np.concatenate(masked_values, axis=0)

    if plot:
        assert pred_value is not None
        assert characteristics is not None
        assert limits is not None
        assert name is not None
        assert params is not None

        plt.figure(figsize=(16, 8))  # Adjust the figure size as needed

        for i in range(8):
            # Filter out masked values
            non_mask_indices = masked_values[:, i] == 1
            non_mask_true_values = true_values[non_mask_indices, i]
            non_mask_predicted_values = predicted_values[non_mask_indices, i]

            plt.subplot(2, 4, i + 1)  # 2 rows, 4 columns, i+1 subplot index
            plt.scatter(non_mask_true_values, non_mask_predicted_values)
            plt.xlabel("True Value")
            plt.ylabel("Predicted Value")
            plt.title(f"{i + 1}'th 3hrs of the day ({characteristics} to {pred_value})")

            min_value, max_value = limits[0], limits[1]

            # Adjust the limits based on the threshold
            range_value = max_value - min_value
            threshold = 0.05
            min_value -= threshold * range_value
            max_value += threshold * range_value

            # Set limits of both x-axis and y-axis based on the adjusted minimum and maximum values
            plt.xlim(min_value, max_value)
            plt.ylim(min_value, max_value)

        plt.tight_layout()
        plt.savefig(f"figures/{name}_{params}_{characteristics}_to_{pred_value}", dpi=300)

    return total_loss / len(dataloader)


def train(model, train_loader, val_loader, checkpoint_pth, criterion, task, learning_rate, epochs, patience):

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
        for (X, masks_X, observed_tp), (y, mask_y) in train_loader:
            X, masks_X, y = X.to(device), masks_X.to(device), y.to(device)
            out = model(X, observed_tp, masks_X)
            loss = criterion(out, y, mask_y)

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



def train_and_eval_regr(X_cols, y_cols, params, task, sequence_length, characteristics, limits):
    torch.manual_seed(1505)
    np.random.seed(1505)
    torch.cuda.manual_seed(1505)

    validation_set_percentage = 0.3

    epochs = 200
    patience = 20

    # Parameters:
    num_heads = params["num_heads"]
    rec_hidden = params["rec_hidden"]
    embed_time = params["embed_time"]
    batch_size = params["batch_size"]
    lr = params["lr"]

    params_print = (f"num_heads={num_heads}, rec_hidden={rec_hidden}, embed_time={embed_time}, "
                    f"sequence_length={sequence_length}, batch_size={batch_size}")

    pvt_cols = ["DATETIME"] + X_cols + y_cols

    dim = len(X_cols)

    df_path_train = "data/training_set_classif.csv"
    df_pvt_train, mean_stds = load_df(df_path=df_path_train, pvt_cols=pvt_cols, parse_dates=["DATETIME"],
                                     normalize=True, y_cols=y_cols)

    df_path_test = "data/test_set_classif.csv"
    df_pvt_test, _ = load_df(df_path=df_path_test, pvt_cols=pvt_cols, parse_dates=["DATETIME"], normalize=True,
                            stats=mean_stds, y_cols=y_cols)

    df_pvt_train.to_csv("data/pvt_df_train.csv")
    df_pvt_test.to_csv("data/pvt_df_test.csv")

    train_df = pd.read_csv("data/pvt_df_train.csv", parse_dates=['DATETIME'], index_col='DATETIME')
    test_df = pd.read_csv("data/pvt_df_test.csv", parse_dates=['DATETIME'], index_col='DATETIME')

    trainX = train_df[X_cols]
    means_X = trainX.mean(skipna=True)

    # Create a dataset and dataloader
    training_dataset = TimeSeriesDataset(dataframe=train_df, sequence_length=sequence_length,
                                         X_cols=X_cols, y_cols=y_cols, final_train=True, means_X=means_X)

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
                        embed_time=embed_time, num_heads=num_heads, device=device, output_len=sequence_length).to(device)
    # Loss
    #criterion = MaskedMSELoss()
    criterion = MaskedSmoothL1Loss()

    # Train the model
    training_loss, validation_loss = train(model=model, train_loader=train_loader, val_loader=val_loader,
                                           checkpoint_pth=None, criterion=criterion, task=task, learning_rate=lr,
                                           epochs=epochs, patience=patience)
    
    print(f'Final Training Loss : {training_loss:.6f} &  Validation Loss : {validation_loss:.6f}\n')

    # Create a dataset and dataloader
    testing_dataset_sliding = TimeSeriesDataset(dataframe=test_df, sequence_length=sequence_length,
                                                X_cols=X_cols, y_cols=y_cols, means_X=means_X)
    testing_dataset_per_day = TimeSeriesDataset(dataframe=test_df, sequence_length=sequence_length,
                                                X_cols=X_cols, y_cols=y_cols, final_train=True, per_day=True,
                                                means_X=means_X)
    training_dataset_per_day = TimeSeriesDataset(dataframe=train_df, sequence_length=sequence_length,
                                                 X_cols=X_cols, y_cols=y_cols, per_day=True, means_X=means_X)

    test_loader_sliding = DataLoader(testing_dataset_sliding, batch_size=batch_size, shuffle=False)
    test_loader_per_day = DataLoader(testing_dataset_per_day, batch_size=batch_size, shuffle=False)
    train_loader_per_day = DataLoader(training_dataset_per_day, batch_size=batch_size, shuffle=False)

    print("Test dataloaders loaded")

    trained_model = MtanGruRegr(input_dim=dim, query=torch.linspace(0, 1., embed_time), nhidden=rec_hidden,
                        embed_time=embed_time, num_heads=num_heads, device=device, output_len=sequence_length).to(device)
    checkpoint = torch.load(f'best_model_{task}.pth')
    trained_model.load_state_dict(checkpoint['mod_state_dict'])

    # Test model's performance on unseen data
    testing_loss = evaluate(trained_model, test_loader_sliding, criterion, plot=True, pred_value=y_cols[0], limits=limits,
                            characteristics=characteristics, params=params_print, name="test_sliding_win")
    print(f'Testing Loss (SmoothL1Loss) for sliding window : {testing_loss:.6f}')

    testing_loss = evaluate(trained_model, test_loader_per_day, criterion, plot=True, pred_value=y_cols[0], limits=limits,
                            characteristics=characteristics, params=params_print, name="test_daily")
    print(f'Testing Loss (SmoothL1Loss) daily : {testing_loss:.6f}')

    training_loss = evaluate(trained_model, train_loader, criterion, plot=True, pred_value=y_cols[0], limits=limits,
                            characteristics=characteristics, params=params_print, name="train_sliding_win")
    print(f'Training Loss (SmoothL1Loss) for sliding window : {training_loss:.6f}')

    training_loss = evaluate(trained_model, train_loader_per_day, criterion, plot=True, pred_value=y_cols[0], limits=limits,
                            characteristics=characteristics, params=params_print, name="train_daily")
    print(f'Training Loss (SmoothL1Loss) daily : {training_loss:.6f}')


def main_loop():
    print("Weather -> QPVT\n")

    sequence_length = 24 // 3
    X_cols = ["humidity", "pressure", "feels_like", "temp", "wind_speed"]
    y_cols = ["Q_PVT"]
    params = {'batch_size': 32, 'lr': 0.001, 'num_heads': 2, 'rec_hidden': 64, 'embed_time': sequence_length}
    task = "day_weather_to_day_qpvt"

    train_and_eval_regr(X_cols=X_cols, y_cols=y_cols, params=params, task=task, sequence_length=sequence_length,
                   characteristics="weather", limits = (-1., 8.5))

    print("Weather -> PYRANOMETER\n")

    sequence_length = 24 // 3
    X_cols = ["humidity", "pressure", "feels_like", "temp", "wind_speed"]
    y_cols = ["PYRANOMETER"]
    params = {'batch_size': 32, 'lr': 0.001, 'num_heads': 2, 'rec_hidden': 64, 'embed_time': sequence_length}
    task = "day_weather_to_day_pyran"

    train_and_eval_regr(X_cols=X_cols, y_cols=y_cols, params=params, task=task, sequence_length=sequence_length,
                        characteristics="weather", limits = (-0.1, 1.1))

    print("PYRANOMETER -> QPVT\n")

    sequence_length = 24 // 3
    X_cols = ["PYRANOMETER"]
    y_cols = ["Q_PVT"]
    params = {'batch_size': 8, 'lr': 0.001, 'num_heads': 2, 'rec_hidden': 8, 'embed_time': 32}
    task = "day_pyran_to_day_qpvt"

    train_and_eval_regr(X_cols=X_cols, y_cols=y_cols, params=params, task=task, sequence_length=sequence_length,
                        characteristics="PYRANOMETER", limits = (-1., 8.5))