import logging
from torch.utils.data import DataLoader
from .loader import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s:%(lineno)d:%(levelname)s:%(name)s:%(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f'Device is {device}')

def main():
    path = "data/DATA_FROM_PLC.csv"
    hp, pv = data["hp"], data["pv"]
    func = lambda x: x.mean()

    df = load(path=path, parse_dates=["Date&time"], normalize=True, grp="30min", agg=func, hist_data=True)
    
    # HP SYSTEM ####################################################

    df_hp = prepare(df, system="hp")

    ds_hp = TSDataset(dataframe=df_hp, seq=15, X=hp["X"], y=hp["y"])
    ds_train_hp, ds_valid_hp = split(ds_hp, vperc=0.2)

    dl_train_hp = DataLoader(ds_train_hp, batch_size=16, shuffle=True)
    dl_valid_hp = DataLoader(ds_valid_hp, batch_size=16, shuffle=False)

    #model = ... .to(device)

    # PV SYSTEM ####################################################

    df_pv = prepare(df, system="pv")

    ds_pv = TSDataset(dataframe=df_pv, seq=15, X=pv["X"], y=pv["y"])
    ds_train_pv, ds_valid_pv = split(ds_pv, vperc=0.2)

    dl_train_pv = DataLoader(ds_train_pv, batch_size=16, shuffle=True)
    dl_valid_pv = DataLoader(ds_valid_pv, batch_size=16, shuffle=False)

    #model = ... .to(device)