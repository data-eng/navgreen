import time
import numpy as np
import warnings
import logging
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from .model import InterpClassif
from utils import CrossEntropyLoss, get_prfs, get_path, save_json, load_json, visualize, tensor_to_python_numbers, save_csv
from .data_loader import load_df, TimeSeriesDataset

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s:%(lineno)d:%(levelname)s:%(name)s:%(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(model, dataloader, criterion, seed, plot=False, pred_value=None, set_type="Train"):
    model.eval()
    total_loss = 0
    true_values, predicted_values = [], []
    true_values_all, predicted_values_all = [], []
    masked_values = []

    for X, (y, masks_y) in dataloader:
        X, y, masks_y = X.to(device), y.to(device), masks_y.to(device)

        with torch.no_grad():
            out = model(X)
            total_loss += criterion(out, y, masks_y)

            if plot:
                if out.shape[0] != X.shape[0]:
                    out = out.unsqueeze(0)

                # Append masked true and predicted values
                true_values.append(y.int().cpu().numpy())
                predicted_values.append(out.cpu().numpy())
                masked_values.append(masks_y.cpu().numpy())

    prfs = None
    predicted_values_probs = None
    if plot:
        true_values = np.concatenate(true_values, axis=0)
        predicted_values = np.concatenate(predicted_values, axis=0)
        masked_values = np.concatenate(masked_values, axis=0)

        assert pred_value is not None

        if predicted_values.ndim == 2: predicted_values = np.transpose(predicted_values)
        # Apply softmax along the last dimension
        predicted_values = np.exp(predicted_values) / np.sum(np.exp(predicted_values), axis=-1, keepdims=True)
        predicted_values_probs = predicted_values.copy()
        # Get the index of the maximum probability along the last dimension
        predicted_values = np.argmax(predicted_values, axis=-1)

        true_values_all = true_values
        predicted_values_all = predicted_values

        non_mask_indices = masked_values == 1
        true_values = true_values[non_mask_indices]
        predicted_values = predicted_values[non_mask_indices]

        # Compute confusion matrix
        true_values, predicted_values = true_values.flatten(), predicted_values.flatten()
        true_values_all, predicted_values_all = true_values_all.flatten(), predicted_values_all.flatten()
        predicted_values_probs = predicted_values_probs.reshape(-1, predicted_values_probs.shape[-1])

        visualize(type="heatmap",
                        values=(true_values, predicted_values),
                        labels=("True Values", "Predicted Values"),
                        title=f"{set_type} Heatmap "+pred_value,
                        classes=["0", "1", "2", "3", "4"],
                        coloring=['azure', 'darkblue'],
                        path=get_path(dirs=["models", pred_value, "interpolation", str(seed)]))

        # Compute scores
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)

            prfs = get_prfs(true_values, predicted_values)
            '''
            logger.info(f"Micro    | f1 score: {prfs['fscore_micro']:.6f} & precision {prfs['precision_micro']:.6f} & recall {prfs['recall_micro']:.6f}")
            logger.info(f"Macro    | f1 score: {prfs['fscore_macro']:.6f} & precision {prfs['precision_macro']:.6f} & recall {prfs['recall_macro']:.6f}")
            logger.info(f"Weighted | f1 score: {prfs['fscore_weighted']:.6f} & precision {prfs['precision_weighted']:.6f} & recall {prfs['recall_weighted']:.6f}")
            '''
    return total_loss / len(dataloader), prfs, (true_values_all, predicted_values_all, predicted_values_probs)


def train(model, train_loader, val_loader, criterion, learning_rate, epochs, patience, seed, pred_value):
    checkpoints = {'epochs': 0, 'best_epoch': 0, 'best_train_loss': float('inf'),
                   'best_val_loss': float('inf')}

    # Early stopping variables
    best_val_loss = float('inf')
    final_train_loss = float('inf')
    epochs_without_improvement = 0
    train_losses, val_losses = [], []

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0

        start_time = time.time()
        for X, (y, masks_y) in train_loader:
            X, y, masks_y = X.to(device), y.to(device), masks_y.to(device)
            out = model(X)
            loss = criterion(out, y, masks_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Get validation loss
        val_loss, _, _ = evaluate(model, val_loader, criterion, seed=seed)
        # Compute average training loss
        average_loss = total_loss / len(train_loader)
        #logger.info(f'Epoch {epoch} | Training Loss: {average_loss:.6f}, Validation Loss: {val_loss:.6f}, '
        #      f'Time : {(time.time() - start_time) / 60:.2f} minutes')
        '''
        if epoch % 50 == 0:
            logger.info(f'Epoch {epoch} | Best training Loss: {final_train_loss:.6f}, Best validation Loss: {best_val_loss:.6f}')
        '''
        train_losses.append(average_loss)
        val_losses.append(val_loss.cpu())

        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            final_train_loss = average_loss
            epochs_without_improvement = 0

            mfn = get_path(dirs=["models", pred_value, "interpolation", str(seed)], name="interpolation.pth")
            torch.save(model.state_dict(), mfn)
            checkpoints.update({'best_epoch': epoch, 'best_train_loss': final_train_loss, 'best_val_loss': best_val_loss})

        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
           # logger.info(f"Early stopping after {epoch} epochs without improvement. Patience is {patience}.")
            break

        checkpoints.update({'epochs': epoch})
    #logger.info("Training complete!")

    cfn = get_path(dirs=["models", pred_value, "interpolation", str(seed)], name="train_losses.json")
    save_json(data=train_losses, filename=cfn)

    visualize(type="multi-plot", values=[(range(1, len(train_losses) + 1), train_losses),
                                         (range(1, len(val_losses) + 1), val_losses)],
              labels=("Epoch", "Loss"), title="Loss Curves", plot_func=plt.plot, coloring=['brown', 'royalblue'],
              names=["Training", "Validation"], path=get_path(dirs=["models", pred_value, "interpolation", str(seed)]))

    return final_train_loss, best_val_loss, checkpoints


def train_model(X_cols, y_cols, params, sequence_length, interpolation, weights, seed=1505):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

    validation_set_percentage = 0.2

    epochs = 1000
    patience = 200

    # Parameters:
    batch_size = params["batch_size"]
    lr = params["lr"]
    pred_value = y_cols[0]
    pvt_cols = ["DATETIME"] + X_cols + y_cols

    dim = len(X_cols)

    train_df, mean_stds = load_df(df_path="../../../data/training_set_classif_new_classes.csv", pvt_cols=pvt_cols, parse_dates=["DATETIME"],
                                      normalize=True, y_cols=y_cols)
    save_json(mean_stds, 'interpolation/mean_stds.json')

    train_df.to_csv("../../../data/pvt_df_train.csv")
    train_df = pd.read_csv("../../../data/pvt_df_train.csv", parse_dates=['DATETIME'], index_col='DATETIME')

    # Create a dataset and dataloader
    training_dataset = TimeSeriesDataset(dataframe=train_df, sequence_length=sequence_length,
                                         X_cols=X_cols, y_cols=y_cols, interpolation=interpolation)

    # Get the total number of samples and compute size of each corresponding set
    total_samples = len(training_dataset)
    validation_size = int(validation_set_percentage * total_samples)
    train_size = total_samples - validation_size

    # Use random_split to create training and validation datasets
    train_dataset, val_dataset = random_split(training_dataset, [train_size, validation_size])

    # Create dataloaders for training and validation sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    # Configure model
    model = InterpClassif(dim=dim, init_embed=sequence_length).to(device)
    # Loss
    criterion = CrossEntropyLoss(weights=torch.tensor(weights).to(device))
    #criterion = CrossEntropyLoss(weights=torch.tensor([0.75, 0.055, 0.02, 0.035, 0.14]).to(device))

    # Train the model
    training_loss, validation_loss, checkpoints = train(model=model, train_loader=train_loader, val_loader=val_loader,
                                           criterion=criterion, learning_rate=lr, epochs=epochs, patience=patience, seed=seed, pred_value=pred_value)
    checkpoints['seed'] = seed

    #logger.info(f'Final Training Loss : {training_loss:.6f} &  Validation Loss : {validation_loss:.6f}\n')


    train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    trained_model = InterpClassif(dim=dim, init_embed=sequence_length).to(device)

    mfn = get_path(dirs=["models", pred_value, "interpolation", str(seed)], name="interpolation.pth")
    trained_model.load_state_dict(torch.load(mfn))

    _, prfs, _ = evaluate(trained_model, train_loader, criterion, plot=True, pred_value=pred_value, seed=seed)

    checkpoints.update(**prfs)
    cfn = get_path(dirs=["models", pred_value, "interpolation", str(seed)], name="train_checkpoints.json")
    save_json(data=tensor_to_python_numbers(checkpoints), filename=cfn)



def test_model(X_cols, y_cols, sequence_length, interpolation, seed, weights):

    pvt_cols = ["DATETIME"] + X_cols + y_cols
    pred_value = y_cols[0]
    dim = len(X_cols)

    mean_stds = load_json('interpolation/mean_stds.json')
    test_df, _ = load_df(df_path="../../../data/test_set_classif_new_classes.csv", pvt_cols=pvt_cols, parse_dates=["DATETIME"], normalize=True,
                         stats=mean_stds, y_cols=y_cols)

    test_df.to_csv("../../../data/pvt_df_test.csv")
    test_df = pd.read_csv("../../../data/pvt_df_test.csv", parse_dates=['DATETIME'], index_col='DATETIME')
    # Loss
    criterion = CrossEntropyLoss(weights=torch.tensor(weights).to(device))

    # Create a dataset and dataloader
    testing_dataset_per_day = TimeSeriesDataset(dataframe=test_df, sequence_length=sequence_length, X_cols=X_cols,
                                                y_cols=y_cols, per_day=True, interpolation=interpolation)

    test_loader_per_day = DataLoader(testing_dataset_per_day, batch_size=1, shuffle=False, drop_last=True)

    trained_model = InterpClassif(dim=dim, init_embed=sequence_length).to(device)

    mfn = get_path(dirs=["models", pred_value, "interpolation", str(seed)], name="interpolation.pth")
    trained_model.load_state_dict(torch.load(mfn))

    test_loss, prfs, (true_values, predicted_values, predicted_prob) = (
        evaluate(trained_model, test_loader_per_day, criterion, plot=True, pred_value=pred_value, seed=seed, set_type="Test"))

    df = test_df.copy()
    data = {"DATETIME": df.index,
            **{col: df[col].values for col in X_cols},
            f"{pred_value}_real": true_values,
            f"{pred_value}_pred": predicted_values,
            f"{pred_value}_probs": predicted_prob.tolist()}

    dfn = get_path(dirs=["models", pred_value, "interpolation", str(seed)], name="data.csv")
    save_csv(data=data, filename=dfn)

    checkpoints = {'seed': seed, 'test_loss': test_loss, **prfs}
    cfn = get_path(dirs=["models", pred_value, "interpolation", str(seed)], name="test_checkpoints.json")
    save_json(data=tensor_to_python_numbers(checkpoints), filename=cfn)

def main_loop_train(seed, y_cols, weights):

    sequence_length = 24 // 3

    interpolation = 'linear'

    X_cols = ["humidity", "pressure", "feels_like", "temp", "wind_speed", "rain_1h"]
    params = {'batch_size': 16, 'lr': 0.001}


    train_model(X_cols=X_cols, y_cols=y_cols, params=params, sequence_length=sequence_length,
                interpolation=interpolation, seed=seed, weights=weights)

def main_loop_test(seed, y_cols, weights):

    sequence_length = 24 // 3

    interpolation = 'linear'

    X_cols = ["humidity", "pressure", "feels_like", "temp", "wind_speed", "rain_1h"]
    params = {'batch_size': 16, 'lr': 0.001}

    test_model(X_cols=X_cols, y_cols=y_cols, sequence_length=sequence_length, interpolation=interpolation, seed=seed, weights=weights)


def main():
    weights = [0.75, 0.055, 0.02, 0.035, 0.14]
    main_loop_train(seed=1505, y_cols=["FixedBin"], weights=weights)
    main_loop_test(seed=1505, y_cols=["FixedBin"], weights=weights)
