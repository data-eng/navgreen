import time
import logging
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from hist_data_analysis import utils
from .model import Transformer
from .loader import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s:%(lineno)d:%(levelname)s:%(name)s:%(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f'Device is {device}')

def train(data, num_classes, epochs, patience, lr, criterion, model, optimizer, scheduler, path, plot=False):
    model.to(device)
    train_data, val_data = data
    batches = len(train_data)
    best_val_loss = float('inf')
    ylabel = params["y"][0]
    stationary = 0
    train_losses, val_losses = [], []

    optimizer = utils.get_optim(optimizer, model, lr)
    scheduler = utils.get_sched(*scheduler, optimizer)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        true_values, pred_values = [], []
        start_time = time.time()

        progress_bar = tqdm(enumerate(train_data), total=batches, desc=f"Epoch {epoch + 1}/{epochs}", leave=True)

        for _, (X, y, mask) in progress_bar:
            X, y, mask = X.to(device), utils.hot3D(y, num_classes, device), mask.to(device)
            y_pred = model(X, mask)
            
            loss = criterion(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(Loss=loss.item())

            batch_size, seq_len, features_size = y.size()
            y = y.reshape(batch_size * seq_len, features_size)
            y_pred = y_pred.reshape(batch_size * seq_len, features_size)

            true_values.append(y.cpu().numpy())
            pred_values.append(y_pred.detach().cpu().numpy())

        avg_loss = total_loss / batches
        train_losses.append(avg_loss)

        logger.info(f"Epoch [{epoch + 1}/{epochs}], Training Loss: {avg_loss:.6f}, "
                    f"Time: {(time.time() - start_time) / 60:.2f} minutes")

        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for X, y, mask in val_data:
                X, y, mask = X.to(device), utils.hot3D(y, num_classes, device), mask.to(device)
                y_pred = model(X, mask)

                val_loss = criterion(y_pred, y)
                total_val_loss += val_loss.item()
        
        avg_val_loss = total_val_loss / batches
        val_losses.append(avg_val_loss)

        logger.info(f"Epoch [{epoch + 1}/{epochs}], Validation Loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            stationary = 0
            torch.save(model.state_dict(), path)

            true_values = np.concatenate(true_values)
            pred_values = np.concatenate(pred_values)

            true_classes = [utils.get_max(pred).index for pred in true_values]
            pred_classes = [utils.get_max(pred).index for pred in pred_values]

            if plot:
                utils.visualize(values=(true_classes, pred_classes), 
                        labels=("True Values", "Predicted Values"), 
                        title="train_"+ylabel,
                        color='brown',
                        plot_func=plt.scatter)
        else:
            stationary += 1

        if stationary >= patience:
            logger.info(f"Early stopping after {epoch + 1} epochs without improvement. Patience is {patience}.")
            break

        scheduler.step() 
    
    if plot:
        utils.visualize(values=[(range(1, len(train_losses) + 1), train_losses), (range(1, len(val_losses) + 1), val_losses)], 
                        labels=("Epoch", "Loss"), 
                        title="Loss Curves", 
                        names=["Training", "Validation"], 
                        colors=['royalblue', 'olivedrab'],
                        plot_func=plt.plot)

    logger.info("Training complete!")

    return avg_loss, best_val_loss

def main():
    path = "data/owm+plc/training_set_classif.csv"
    seq_len = 1440 // 180
    num_classes = 5
    batch_size = 32

    df = load(path=path, parse_dates=["DATETIME"], normalize=True)
    df_prep = prepare(df, phase="train")

    ds = TSDataset(df=df_prep, seq_len=seq_len, X=params["X"], t=params["t"], y=params["y"])
    ds_train, ds_val = split(ds, vperc=0.2)

    dl_train = DataLoader(ds_train, batch_size, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size, shuffle=False)

    """ # The are no nans!
    for batch_idx, (X, y, _) in enumerate(dl_val):
        for seq in X:
            if(torch.isnan(seq).any()):
               print(seq)
    """

    train_loss, val_loss = train(data=(dl_train, dl_val),
                                 num_classes=num_classes,
                                 epochs=10,
                                 patience=2,
                                 lr=1e-4,
                                 criterion=nn.CrossEntropyLoss(),
                                 model=Transformer(in_size=len(params["X"])+len(params["t"]), out_size=num_classes),
                                 optimizer="AdamW",
                                 scheduler=("StepLR", 1.0, 0.98),
                                 path="models/transformer.pth",
                                 plot=True
                                 )
    logger.info(f'Final Training Loss : {train_loss:.6f} & Validation Loss : {val_loss:.6f}\n')