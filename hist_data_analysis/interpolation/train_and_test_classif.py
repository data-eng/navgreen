import time
import numpy as np
import pandas as pd
import warnings

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, precision_recall_curve, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from .model import InterpClassif
from .utils import CrossEntropyLoss
from .data_loader import load_df, TimeSeriesDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def class_wise_pr_roc(labels, predicted_labels, name):
    num_classes = len(np.unique(labels))
    predicted_probs = label_binarize(predicted_labels, classes=np.arange(num_classes))
    precision = dict()
    recall = dict()
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    pr_auc = dict()

    for i in range(num_classes):
        # Compute precision and recall for each class
        precision[i], recall[i], _ = precision_recall_curve(labels == i, predicted_probs[:, i])
        pr_auc[i] = auc(recall[i], precision[i])

        # Compute ROC curve for each class
        fpr[i], tpr[i], _ = roc_curve(labels == i, predicted_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

        # Plot PR curve
        plt.figure()
        plt.plot(recall[i], precision[i], lw=2, label='PR curve (area = %0.2f)' % pr_auc[i])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall curve for class {i}')
        plt.legend(loc="lower left")
        plt.savefig(f'figures/pr_class_{i}_{name}.png', dpi=300)

        # Plot ROC curve
        plt.figure()
        plt.plot(fpr[i], tpr[i], lw=2, label='ROC curve (area = %0.2f)' % roc_auc[i])
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC curve for class {i}')
        plt.legend(loc="lower right")
        plt.savefig(f'figures/roc_class_{i}_{name}.png', dpi=300)


def evaluate(model, dataloader, criterion, plot=False, pred_value=None, characteristics=None, name=None,
             params=None):
    model.eval()
    total_loss = 0
    true_values = []
    predicted_values = []
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

    if plot:
        true_values = np.concatenate(true_values, axis=0)
        predicted_values = np.concatenate(predicted_values, axis=0)
        masked_values = np.concatenate(masked_values, axis=0)

        assert pred_value is not None
        assert characteristics is not None
        assert name is not None
        assert params is not None

        if predicted_values.ndim == 2: predicted_values = np.transpose(predicted_values)
        # Apply softmax along the last dimension
        predicted_values = np.exp(predicted_values) / np.sum(np.exp(predicted_values), axis=-1, keepdims=True)
        # Get the index of the maximum probability along the last dimension
        predicted_values = np.argmax(predicted_values, axis=-1)

        non_mask_indices = masked_values == 1
        true_values = true_values[non_mask_indices]
        predicted_values = predicted_values[non_mask_indices]

        # Compute confusion matrix
        true_values, predicted_values = true_values.flatten(), predicted_values.flatten()
        cm = confusion_matrix(true_values, predicted_values)

        class_labels = ["< 0.42 KWh", "< 1.05 KWh", "< 1.51 KWh", "< 2.14 KWh", ">= 2.14 KWh"]
        # Plot confusion matrix as heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', xticklabels=class_labels, yticklabels=class_labels)
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix')
        plt.savefig(f"figures/cm_{name}_{params}_{characteristics}_to_{pred_value}", dpi=300)

        # Compute scores
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)

            precision, recall, f1, _ = precision_recall_fscore_support(true_values, predicted_values, average='micro')
            print(f"Micro    | f1 score: {f1:.6f} & precision {precision:.6f} & recall {recall:.6f}")

            precision, recall, f1, _ = precision_recall_fscore_support(true_values, predicted_values, average='macro')
            print(f"Macro    | f1 score: {f1:.6f} & precision {precision:.6f} & recall {recall:.6f}")

            precision, recall, f1, _ = precision_recall_fscore_support(true_values, predicted_values,
                                                                       average='weighted')
            print(f"Weighted | f1 score: {f1:.6f} & precision {precision:.6f} & recall {recall:.6f}")

        class_wise_pr_roc(labels=true_values, predicted_labels=predicted_values,
                          name=f'{name}_{params}_{characteristics}_to_{pred_value}')

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
        for X, (y, masks_y) in train_loader:
            X, y, masks_y = X.to(device), y.to(device), masks_y.to(device)
            out = model(X)
            loss = criterion(out, y, masks_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Get validation loss
        val_loss = evaluate(model, val_loader, criterion)
        # Compute average training loss
        average_loss = total_loss / len(train_loader)
        #print(f'Epoch {epoch} | Training Loss: {average_loss:.6f}, Validation Loss: {val_loss:.6f}, '
        #      f'Time : {(time.time() - start_time) / 60:.2f} minutes')

        if epoch % 50 == 0:
            print(f'Epoch {epoch} | Best training Loss: {final_train_loss:.6f}, Best validation Loss: {best_val_loss:.6f}')

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


def train_and_eval(X_cols, y_cols, params, task, sequence_length, characteristics, interpolation):
    torch.manual_seed(1505)
    np.random.seed(1505)
    torch.cuda.manual_seed(1505)

    validation_set_percentage = 0.3

    epochs = 10 #400
    patience = 50

    # Parameters:
    batch_size = params["batch_size"]
    lr = params["lr"]

    params_print = f"sequence_length={sequence_length}, batch_size={batch_size}"

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
    #criterion = CrossEntropyLoss(weights=torch.tensor([0.25, 0.2, 0.15, 0.2, 0.2]).to(device))
    criterion = CrossEntropyLoss(weights=torch.tensor([0.75, 0.055, 0.02, 0.035, 0.14]).to(device))

    # Train the model
    training_loss, validation_loss = train(model=model, train_loader=train_loader, val_loader=val_loader,
                                           checkpoint_pth=None, criterion=criterion, task=task, learning_rate=lr,
                                           epochs=epochs, patience=patience)

    print(f'Final Training Loss : {training_loss:.6f} &  Validation Loss : {validation_loss:.6f}\n')

    # Create a dataset and dataloader

    testing_dataset_sliding = TimeSeriesDataset(dataframe=test_df, sequence_length=sequence_length,
                                                X_cols=X_cols, y_cols=y_cols, interpolation=interpolation)
    testing_dataset_per_day = TimeSeriesDataset(dataframe=test_df, sequence_length=sequence_length, X_cols=X_cols,
                                                y_cols=y_cols, per_day=True, interpolation=interpolation)
    training_dataset_per_day = TimeSeriesDataset(dataframe=train_df, sequence_length=sequence_length,
                                                 X_cols=X_cols, y_cols=y_cols, per_day=True, interpolation=interpolation)

    test_loader_sliding = DataLoader(testing_dataset_sliding, batch_size=1, shuffle=False, drop_last=True)
    test_loader_per_day = DataLoader(testing_dataset_per_day, batch_size=1, shuffle=False, drop_last=True)
    train_loader_per_day = DataLoader(training_dataset_per_day, batch_size=batch_size, shuffle=False, drop_last=True)

    train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    print("Test dataloaders loaded")

    trained_model = InterpClassif(dim=dim, init_embed=sequence_length).to(device)

    checkpoint = torch.load(f'best_model_{task}.pth')
    trained_model.load_state_dict(checkpoint['mod_state_dict'])

    # Test model's performance on unseen data
    testing_loss = evaluate(trained_model, test_loader_sliding, criterion, plot=True, pred_value=y_cols[0],
                            characteristics=characteristics, params=params_print, name="test_sliding_win")
    print(f'Testing Loss (Cross Entropy) for sliding window : {testing_loss:.6f}')

    testing_loss = evaluate(trained_model, test_loader_per_day, criterion, plot=True, pred_value=y_cols[0],
                            characteristics=characteristics, params=params_print, name="test_daily")
    print(f'Testing Loss (Cross Entropy) daily : {testing_loss:.6f}')

    training_loss = evaluate(trained_model, train_loader, criterion, plot=True, pred_value=y_cols[0],
                             characteristics=characteristics, params=params_print, name="train_sliding_win")
    print(f'Training Loss (Cross Entropy) for sliding window : {training_loss:.6f}')

    training_loss = evaluate(trained_model, train_loader_per_day, criterion, plot=True, pred_value=y_cols[0],
                             characteristics=characteristics, params=params_print, name="train_daily")
    print(f'Training Loss (Cross Entropy) daily : {training_loss:.6f}')


def main_loop():
    sequence_length = 24 // 3

    # interpolations = ['linear', 'quadratic', 'cubic', 'slinear', 'nearest', 'polynomial']
    interpolation = 'linear'

    print("Weather -> QPVT")

    X_cols = ["humidity", "pressure", "feels_like", "temp", "wind_speed", "rain_1h"]
    y_cols = ["binned_Q_PVT"]
    params = {'batch_size': 64, 'lr': 0.001}

    task = f"interpol_{interpolation}_day_weather_to_binned_qpvt"
    train_and_eval(X_cols=X_cols, y_cols=y_cols, params=params, task=task, sequence_length=sequence_length,
                   characteristics="weather", interpolation=interpolation)
    '''
    print("\nPYRANOMETER -> QPVT")

    X_cols = ["PYRANOMETER"]
    y_cols = ["binned_Q_PVT"]
    params = {'batch_size': 64, 'lr': 0.001}

    task = f"interpol_{interpolation}_day_PYRANOMETER_to_binned_qpvt"
    train_and_eval(X_cols=X_cols, y_cols=y_cols, params=params, task=task, sequence_length=sequence_length,
                   characteristics="weather", interpolation=interpolation)
    '''

