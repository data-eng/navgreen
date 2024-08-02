import os
from torch.utils.data import DataLoader
from .model import Transformer
from .loader import *

import public_weather_installation.utils as utils

device = "cpu" # Only cpu on the vm


def test(data, df, classes, model, mfn):
    model.load_state_dict(torch.load(mfn, map_location=torch.device('cpu')))

    model.to(device)
    model.eval()
    num_classes = len(classes)

    pred_values = []

    with torch.no_grad():
        for _, (X, mask_X) in enumerate(data):
            X, mask_X = X.to(device), mask_X.to(device)
            y_pred = model(X, mask_X)

            batch_size, seq_len, _ = y_pred.size()
            y_pred = y_pred.reshape(batch_size * seq_len, num_classes)

            pred_values.append(y_pred.cpu().numpy())

    pred_values = np.concatenate(pred_values)
    pred_classes = [utils.get_max(pred).index for pred in pred_values]

    predicted_values_prob = np.exp(pred_values) / np.sum(np.exp(pred_values), axis=-1, keepdims=True)
    data = {"DATETIME": df.index,
            **{col: df[col].values for col in params["X"]},
            f"predicted": pred_classes,
            f"probabilities": predicted_values_prob.tolist()}

    filename = "weather_predictions/communication_PLC/public_weather_installation/model_pred_data.csv"
    if os.path.exists(filename):
        utils.save_csv(data=data, filename=filename, append=True)
    else:
        utils.save_csv(data=data, filename=filename)

    return data


def main_loop(df, model_pth):
    seq_len = 8
    batch_size = 1
    classes = ["0", "1", "2", "3", "4"]

    df = load(df=df, normalize=True)
    df_prep = prepare(df, phase="test")

    ds_test = TSDataset(df=df_prep, seq_len=seq_len, X=params["X"], t=params["t"], per_day=True)
    dl_test = DataLoader(ds_test, batch_size, shuffle=False)

    model = Transformer(in_size=len(params["X"])+len(params["t"]),
                        out_size=len(classes),
                        nhead=1,
                        num_layers=1,
                        dim_feedforward=2048,
                        dropout=0)

    return test(data=dl_test, df=df_prep, classes=classes, model=model, mfn=model_pth)

