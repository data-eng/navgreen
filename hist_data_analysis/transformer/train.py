from loader import *

def main():
    path = "data/DATA_FROM_PLC.csv"
    hp, pv = data["hp"], data["pv"]

    df = load(path=path, parse_dates=["Date&time"], normalize=True, grp="30T", agg=np.mean, hist_data=True)
    df_hp = prepare(df, system="hp")
    ds_hp = TSDataset(dataframe=df_hp, seq=15, X=hp["X"], Y=hp["y"])

    # ...

    df_pv = prepare(df, system="pv")
    ds_pv = TSDataset(dataframe=df_pv, seq=15, X=pv["X"], Y=pv["y"])

    # ...