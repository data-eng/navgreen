import time
import logging
import torch
import matplotlib.pyplot as plt
from torch.nn import MSELoss
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

def train(data, epochs, patience, lr, criterion, model, optimizer, scheduler, path):
    model.to(device)
    train_data, val_data = data
    batches = len(train_data)
    best_val_loss = float('inf')
    stationary = 0
    train_losses, val_losses = [], []

    optimizer = utils.get_optim(optimizer, model, lr)
    scheduler = utils.get_sched(*scheduler, optimizer)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        start_time = time.time()

        progress_bar = tqdm(enumerate(train_data), total=batches, desc=f"Epoch {epoch + 1}/{epochs}", leave=True)

        for _, (X, y) in progress_bar:
            X, y = X.to(device), y.to(device)

            y_pred = model(X)
            
            loss = criterion(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(Loss=loss.item())

        avg_loss = total_loss / batches
        train_losses.append(avg_loss)

        logger.info(f"Epoch [{epoch + 1}/{epochs}], Training Loss: {avg_loss:.6f}, "
                    f"Time: {(time.time() - start_time) / 60:.2f} minutes")

        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for X, y in val_data:
                X, y = X.to(device), y.to(device)
                y_pred = model(X)
            
                val_loss = criterion(y_pred, y)
                total_val_loss += val_loss.item()
        
        avg_val_loss = total_val_loss / batches
        val_losses.append(avg_val_loss)

        logger.info(f"Epoch [{epoch + 1}/{epochs}], Validation Loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            stationary = 0
            torch.save(model.state_dict(), path)
        else:
            stationary += 1

        if stationary >= patience:
            logger.info(f"Early stopping after {epoch + 1} epochs without improvement. Patience is {patience}.")
            break

        scheduler.step() 
    
    utils.visualize(values=[(range(1, len(train_losses) + 1), train_losses), (range(1, len(val_losses) + 1), val_losses)], 
                    labels=("Epoch", "Loss"), 
                    title="Loss Curves", 
                    names=["Training", "Validation"], 
                    colors=['royalblue', 'olivedrab'],
                    plot_func=plt.plot)

    logger.info("Training complete!")

    return avg_loss, best_val_loss

def main():
    path = "data/training_set_before_conv.csv"
    hp, pv = data["hp"], data["pv"]

    batch_size=120
    day_dur, group_dur = 1440, 120
    num_pairs = day_dur // group_dur
    func = lambda x: x.mean()

    df = load(path=path, parse_dates=["Date&time"], normalize=True, grp=f"{group_dur}min", agg=func, hist_data=True)
    
    df_hp = prepare(df, phase="train", system="hp")

    ds_hp = TSDataset(dataframe=df_hp, seq_len=num_pairs, X=hp["X"], y=hp["y"])
    ds_train_hp, ds_val_hp = split(ds_hp, vperc=0.2)

    dl_train_hp = DataLoader(ds_train_hp, batch_size, shuffle=True)
    dl_val_hp = DataLoader(ds_val_hp, batch_size, shuffle=False)

    train_loss_hp, val_loss_hp = train(data=(dl_train_hp, dl_val_hp),
                                       epochs=300,
                                       patience=4,
                                       lr=1e-3,
                                       criterion=MSELoss(),
                                       model=Transformer(in_size=len(hp["X"]), out_size=len(hp["y"])),
                                       optimizer="AdamW",
                                       scheduler=("StepLR", 1.0, 0.98),
                                       path="models/transformer_hp.pth"
                                       )
    logger.info(f'HP | Final Training Loss : {train_loss_hp:.6f} & Validation Loss : {val_loss_hp:.6f}\n')

    df_pv = prepare(df, phase="train", system="pv")

    ds_pv = TSDataset(dataframe=df_pv, seq_len=num_pairs, X=pv["X"], y=pv["y"])
    ds_train_pv, ds_val_pv = split(ds_pv, vperc=0.2)

    dl_train_pv = DataLoader(ds_train_pv, batch_size, shuffle=True)
    dl_val_pv = DataLoader(ds_val_pv, batch_size, shuffle=False)

    train_loss_pv, val_loss_pv = train(data=(dl_train_pv, dl_val_pv),
                                       epochs=300,
                                       patience=4,
                                       lr=1e-3,
                                       criterion=MSELoss(),
                                       model=Transformer(in_size=len(pv["X"]), out_size=len(pv["y"])),
                                       optimizer="AdamW",
                                       scheduler=("StepLR", 1.0, 0.98),
                                       path="models/transformer_pv.pth"
                                       )
    logger.info(f'PV | Final Training Loss : {train_loss_pv:.6f} & Validation Loss : {val_loss_pv:.6f}')