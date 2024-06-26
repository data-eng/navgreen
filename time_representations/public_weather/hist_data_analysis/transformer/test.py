import logging
import torch
from torch.utils.data import DataLoader
# from tqdm import tqdm
import utils
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

def test(data, df, classes, criterion, model, seed, y, visualize=True):
    ylabel = y[0]
    mfn = utils.get_path(dirs=["models", ylabel, "transformer", str(seed)], name="transformer.pth")
    model.load_state_dict(torch.load(mfn))

    model.to(device)
    model.eval()
    batches = len(data)
    num_classes = len(classes)

    total_test_loss = 0.0
    true_values, pred_values = [], []

    checkpoints = {'seed': seed, 
                   'test_loss': float('inf'),
                   'precision_micro': 0, 
                   'precision_macro': 0, 
                   'precision_weighted': 0,
                   'recall_micro': 0, 
                   'recall_macro': 0, 
                   'recall_weighted': 0, 
                   'fscore_micro': 0,
                   'fscore_macro': 0, 
                   'fscore_weighted': 0}

    #progress_bar = tqdm(enumerate(data), total=batches, desc=f"Evaluation", leave=True)
    #logger.info(f"\nTesting with seed {seed} just started...")

    with torch.no_grad():
        #for _, (X, y, mask_X, mask_y) in progress_bar:
        for _, (X, y, mask_X, mask_y) in enumerate(data):
            X, y, mask_X, mask_y = X.to(device), y.long().to(device), mask_X.to(device), mask_y.to(device)
            y_pred = model(X, mask_X)

            batch_size, seq_len, _ = y_pred.size()
            y_pred = y_pred.reshape(batch_size * seq_len, num_classes)
            y = y.reshape(batch_size * seq_len)
            mask_y = mask_y.reshape(batch_size * seq_len)
        
            y_pred = utils.mask(tensor=y_pred, mask=mask_y, id=0)
            y = utils.mask(tensor=y, mask=mask_y, id=0)
            
            test_loss = criterion(pred=y_pred, true=y)
            total_test_loss += test_loss.item()

            true_values.append(y.cpu().numpy())
            pred_values.append(y_pred.cpu().numpy())
    
    avg_test_loss = total_test_loss / batches
    
    true_values = np.concatenate(true_values)
    pred_values = np.concatenate(pred_values)

    true_classes = true_values.tolist()
    pred_classes = [utils.get_max(pred).index for pred in pred_values]

    if visualize:
        utils.visualize(type="heatmap",
                        values=(true_classes, pred_classes), 
                        labels=("True Values", "Predicted Values"), 
                        title="Test Heatmap "+ylabel,
                        classes=classes,
                        coloring=['azure', 'darkblue'],
                        path=utils.get_path(dirs=["models", ylabel, "transformer", str(seed)]))
        
    checkpoints.update({'test_loss': avg_test_loss, **utils.get_prfs(true=true_classes, pred=pred_classes)})
    cfn = utils.get_path(dirs=["models", ylabel, "transformer", str(seed)], name="test_checkpoints.json")
    utils.save_json(data=checkpoints, filename=cfn)

    predicted_values_prob = np.exp(pred_values) / np.sum(np.exp(pred_values), axis=-1, keepdims=True)
    data = {"DATETIME": df.index,
            **{col: df[col].values for col in params["X"]},
            f"{ylabel}_real": true_classes,
            f"{ylabel}_pred": pred_classes,
            f"{ylabel}_probs": predicted_values_prob.tolist()}
    
    dfn = utils.get_path(dirs=["models", ylabel, "transformer", str(seed)], name="data.csv")
    utils.save_csv(data=data, filename=dfn)

    #logger.info(f'\nTesting with seed {seed} complete!\nTesting Loss: {avg_test_loss:.6f}\n')

    return avg_test_loss

def main_loop(seed, y_col):
    path = "../../../data/test_set_classif_new_classes.csv"
    seq_len = 1440 // 180
    batch_size = 1
    classes = ["0", "1", "2", "3", "4"]

    df = load(path=path, parse_dates=["DATETIME"], normalize=True, bin=y_col[0])
    df_prep = prepare(df, phase="test")

    weights = utils.load_json(filename=f'transformer/weights_{y_col[0]}.json')

    ds_test = TSDataset(df=df_prep, seq_len=seq_len, X=params["X"], t=params["t"], y=y_col, per_day=True)
    dl_test = DataLoader(ds_test, batch_size, shuffle=False)

    torch.manual_seed(seed)

    model = Transformer(in_size=len(params["X"])+len(params["t"]),
                        out_size=len(classes),
                        nhead=1,
                        num_layers=1,
                        dim_feedforward=2048,
                        dropout=0)

    _ = test(data=dl_test,
             df=df_prep,
             classes=classes,
             criterion=utils.WeightedCrossEntropyLoss(weights),
             model=model,
             seed=seed,
             y = y_col,
             visualize=True)

def main():
    main_loop(seed=13, y_col="binned_Q_PVT")
