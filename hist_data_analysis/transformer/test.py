import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
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

def test(data, num_classes, criterion, model, path):
    model.load_state_dict(torch.load(path))
    model.to(device)
    batches = len(data)
    total_loss = 0.0
    ylabel = params["y"][0]
    true_values, pred_values = [], []

    progress_bar = tqdm(enumerate(data), total=batches, desc=f"Evaluation", leave=True)

    with torch.no_grad():
        for _, (X, y, mask) in progress_bar:
            X, y, mask = X.to(device), utils.hot3D(y, num_classes, device), mask.to(device)
            y_pred = model(X, mask)
        
            loss = criterion(y_pred, y)
            total_loss += loss.item()

            batch_size, seq_len, features_size = y.size()
            y = y.reshape(batch_size * seq_len, features_size)
            y_pred = y_pred.reshape(batch_size * seq_len, features_size)

            true_values.append(y.cpu().numpy())
            pred_values.append(y_pred.cpu().numpy())
    
    true_values = np.concatenate(true_values)
    pred_values = np.concatenate(pred_values)

    true_classes = [utils.get_max(pred).index for pred in true_values]
    pred_classes = [utils.get_max(pred).index for pred in pred_values]

    utils.visualize(values=(true_classes, pred_classes), 
                    labels=("True Values", "Predicted Values"), 
                    title="test_"+ylabel,
                    color='rebeccapurple',
                    plot_func=plt.scatter)

    avg_loss = total_loss / batches
    logger.info("Evaluation complete!")

    return avg_loss

def main():
    path = "data/owm+plc/test_set_classif.csv"
    seq_len = 1440 // 180
    num_classes = 5
    batch_size=1

    df = load(path=path, parse_dates=["DATETIME"], normalize=True)
    df_prep = prepare(df, phase="test")

    ds_test = TSDataset(df=df_prep, seq_len=seq_len, X=params["X"], t=params["t"], y=params["y"])
    dl_test = DataLoader(ds_test, batch_size, shuffle=False)

    test_loss = test(data=dl_test,
                     num_classes=num_classes,
                     criterion=nn.CrossEntropyLoss(),
                     model=Transformer(in_size=len(params["X"])+len(params["t"]), out_size=num_classes),
                     path="models/transformer.pth"
                     )
    logger.info(f'Evaluation Loss : {test_loss:.6f}\n')