import time
import logging
import torch
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from hist_data_analysis.utils import get_optim, get_sched
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

def test(data, criterion, model, path):
    model.load_state_dict(torch.load(path))
    model.to(device)
    batches = len(data)
    total_loss = 0.0

    progress_bar = tqdm(enumerate(data), total=batches, desc=f"Evaluation", leave=True)

    with torch.no_grad():
        for _, (X, y) in progress_bar:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
        
            loss = criterion(y_pred, y)
            total_loss += loss.item()

    avg_loss = total_loss / batches

    logger.info("Evaluation complete!")

    return avg_loss

def main():
    path = "data/test_set.csv"
    hp, pv = data["hp"], data["pv"]

    batch_size=1
    day_dur, group_dur = 1440, 30
    num_pairs = day_dur // group_dur
    func = lambda x: x.mean()

    df = load(path=path, parse_dates=["DATETIME"], normalize=True, grp=f"{group_dur}min", agg=func, hist_data=True)
    
    df_hp = prepare(df, phase="test", system="hp")
    ds_test_hp = TSDataset(dataframe=df_hp, seq_len=num_pairs, X=hp["X"], y=hp["y"])
    dl_test_hp = DataLoader(ds_test_hp, batch_size, shuffle=False)

    test_loss_hp = test(data=dl_test_hp,
                        criterion=MSELoss(),
                        model=Transformer(in_size=len(hp["X"]), out_size=len(hp["y"])),
                        path="models/transformer_hp.pth"
                        )
    logger.info(f'HP | Evaluation Loss : {test_loss_hp:.6f}\n')

    df_pv = prepare(df, phase="test", system="pv")
    ds_test_pv = TSDataset(dataframe=df_pv, seq_len=num_pairs, X=pv["X"], y=pv["y"])
    dl_test_pv = DataLoader(ds_test_pv, batch_size, shuffle=False)

    test_loss_pv = test(data=dl_test_pv,
                        criterion=MSELoss(),
                        model=Transformer(in_size=len(pv["X"]), out_size=len(pv["y"])),
                        path="models/transformer_pv.pth"
                        )
    logger.info(f'PV | Evaluation Loss : {test_loss_pv:.6f}')