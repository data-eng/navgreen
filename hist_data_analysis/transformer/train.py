import time
import logging
import torch
from torch.nn import MSELoss
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from .loader import *
from .model import Transformer

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s:%(lineno)d:%(levelname)s:%(name)s:%(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f'Device is {device}')

batch_size = 120
epochs = 100
patience = 3
lr = 1e-4
criterion = MSELoss()
model = Transformer().to(device)
optimizer = AdamW(model.parameters(), lr=lr)
scheduler = StepLR(optimizer, 1.0, gamma=0.95)

def train(train_data, val_data):
    batches = len(train_data)
    best_val_loss = float('inf')
    stationary = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        start_time = time.time()

        for X, y in train_data:
            X, y = X.to(device), y.to(device)
            logger.debug(f"X shape: {X.shape}, y shape: {y.shape}")

            y_pred = model(X)
            loss = criterion(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / batches
        logger.info(f"Epoch [{epoch + 1}/{epochs}], Training Loss: {avg_loss:.6f}, "
                    f"Time: {(time.time() - start_time) / 60:.2f} minutes")

        avg_val_loss = evaluate(val_data)
        logger.info(f"Epoch [{epoch + 1}/{epochs}], Validation Loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            stationary = 0
        else:
            stationary += 1

        if stationary >= patience:
            logger.info(f"Early stopping after {epoch + 1} epochs without improvement. Patience is {patience}.")
            break

        scheduler.step() 

    logger.info("Training complete!")

    return avg_loss, best_val_loss

def evaluate(val_data):
    model.eval()
    batches = len(val_data)
    total_loss = 0.0

    with torch.no_grad():
        for X, y in val_data:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
         
            loss = criterion(y_pred, y)
            total_loss += loss.item()

    return total_loss / batches

def main():
    path = "data/DATA_FROM_PLC.csv"
    hp, pv = data["hp"], data["pv"]

    day_dur, group_dur = 1440, 30
    num_pairs = day_dur / group_dur
    func = lambda x: x.mean()

    df = load(path=path, parse_dates=["Date&time"], normalize=True, grp=f"{group_dur}min", agg=func, hist_data=True)
    
    # HP SYSTEM ####################################################

    df_hp = prepare(df, system="hp")

    ds_hp = TSDataset(dataframe=df_hp, len_seq=num_pairs, X=hp["X"], y=hp["y"])
    ds_train_hp, ds_val_hp = split(ds_hp, vperc=0.2)

    seqs_train_hp = create_sequences(ds_train_hp)
    seqs_val_hp = create_sequences(ds_val_hp)

    dl_train_hp = DataLoader(seqs_train_hp, batch_size, shuffle=True)
    dl_val_hp = DataLoader(seqs_val_hp, batch_size, shuffle=False)

    train_loss_hp, val_loss_hp = train(dl_train_hp, dl_val_hp)
    logger.info(f'HP | Final Training Loss : {train_loss_hp:.6f} & Validation Loss : {val_loss_hp:.6f}\n')

    # PV SYSTEM ####################################################

    df_pv = prepare(df, system="pv")

    ds_pv = TSDataset(dataframe=df_pv, len_seq=num_pairs, X=pv["X"], y=pv["y"])
    ds_train_pv, ds_val_pv = split(ds_pv, vperc=0.2)

    seqs_train_pv = create_sequences(ds_train_pv)
    seqs_val_pv = create_sequences(ds_val_pv)

    dl_train_pv = DataLoader(seqs_train_pv, batch_size, shuffle=True)
    dl_val_pv = DataLoader(seqs_val_pv, batch_size, shuffle=False)

    train_loss_pv, val_loss_pv = train(dl_train_pv, dl_val_pv)
    logger.info(f'PV | Final Training Loss : {train_loss_pv:.6f} & Validation Loss : {val_loss_pv:.6f}\n')