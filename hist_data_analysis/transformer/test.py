import logging
import torch
import json
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from hist_data_analysis.transformer import utils
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

def test(data, classes, criterion, model, path, visualize=True):
    model.load_state_dict(torch.load(path))
    model.to(device)
    model.eval()
    batches = len(data)
    num_classes = len(classes)

    total_loss = 0.0
    ylabel = params["y"][0]
    true_values, pred_values = [], []
    results = []

    progress_bar = tqdm(enumerate(data), total=batches, desc=f"Evaluation", leave=True)

    with torch.no_grad():
        for _, (X, y, mask_X, mask_y) in progress_bar:
            X, y, mask_X, mask_y = X.to(device), y.long().to(device), mask_X.to(device), mask_y.to(device)
            y_pred = model(X, mask_X)

            batch_size, seq_len, _ = y_pred.size()
            y_pred = y_pred.reshape(batch_size * seq_len, num_classes)
            y = y.reshape(batch_size * seq_len)
            mask_y = mask_y.reshape(batch_size * seq_len)
        
            y_pred = utils.mask(tensor=y_pred, mask=mask_y, id=0)
            y = utils.mask(tensor=y, mask=mask_y, id=0)
            
            loss = criterion(pred=y_pred, true=y)
            total_loss += loss.item()

            true_values.append(y.cpu().numpy())
            pred_values.append(y_pred.cpu().numpy())
    
    avg_loss = total_loss / batches
    
    true_values = np.concatenate(true_values)
    pred_values = np.concatenate(pred_values)

    true_classes = true_values.tolist()
    pred_classes = [utils.get_max(pred).index for pred in pred_values]

    f1_micro, f1_macro, f1_weighted = utils.get_f1(true=true_classes, pred=pred_classes)
    
    logger.info(f"Testing Loss: {avg_loss:.6f}, F1 Score (Micro) = {f1_micro:.6f}, F1 Score (Macro) = {f1_macro:.6f}, F1 Score (Weighted) = {f1_weighted:.6f}")
    results.append({'test_loss': avg_loss, 'f1_micro': f1_micro, 'f1_macro': f1_macro, 'f1_weighted': f1_weighted})
            
    with open('static/testing_results.json', 'w') as f:
        json.dump(results, f)

    if visualize:
        utils.visualize(type="single-plot",
                        values=(true_classes, pred_classes), 
                        labels=("True Values", "Predicted Values"), 
                        title="Test Scatter Plot "+ylabel,
                        plot_func=plt.scatter,
                        coloring='rebeccapurple',
                        classes=classes,
                        tick=True)
        
        utils.visualize(type="heatmap",
                        values=(true_classes, pred_classes), 
                        labels=("True Values", "Predicted Values"), 
                        title="Test Heatmap "+ylabel,
                        classes=classes)

    logger.info("Evaluation complete!")

    return avg_loss

def main():
    path = "data/owm+plc/test_set_classif.csv"
    seq_len = 1440 // 180
    batch_size = 1
    classes = ["< 0.42 KWh", "< 1.05 KWh", "< 1.51 KWh", "< 2.14 KWh", ">= 2.14 KWh"]

    df = load(path=path, parse_dates=["DATETIME"], normalize=True)
    df_prep = prepare(df, phase="test")

    weights = utils.load_json(filename='static/weights.json')

    ds_test = TSDataset(df=df_prep, seq_len=seq_len, X=params["X"], t=params["t"], y=params["y"])
    dl_test = DataLoader(ds_test, batch_size, shuffle=False)

    model = Transformer(in_size=len(params["X"])+len(params["t"]), 
                        out_size=len(classes),
                        nhead=4, 
                        num_layers=1,
                        dim_feedforward=2048, 
                        dropout=0.1
                        )

    test_loss = test(data=dl_test,
                     classes=classes,
                     criterion=utils.WeightedCrossEntropyLoss(weights),
                     model=model,
                     path="models/transformer.pth",
                     visualize=True
                     )
    
    logger.info(f'Evaluation Loss : {test_loss:.6f}\n')