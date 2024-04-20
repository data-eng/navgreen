import logging
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
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

def train(data, classes, epochs, patience, lr, criterion, model, optimizer, scheduler, seed, visualize=False):
    model.to(device)
    train_data, val_data = data
    batches = len(train_data)
    num_classes = len(classes)
    optimizer = utils.get_optim(optimizer, model, lr)
    scheduler = utils.get_sched(*scheduler, optimizer)
    torch.manual_seed(seed)

    best_val_loss = float('inf')
    ylabel = params["y"][0]
    stationary = 0
    train_losses, val_losses = [], []

    checkpoints = {'seed': seed, 
                   'epochs': 0, 
                   'best_epoch': 0, 
                   'best_train_loss': float('inf'), 
                   'best_val_loss': float('inf'), 
                   'precision_micro': 0, 
                   'precision_macro': 0, 
                   'precision_weighted': 0,
                   'recall_micro': 0, 
                   'recall_macro': 0, 
                   'recall_weighted': 0, 
                   'fscore_micro': 0,
                   'fscore_macro': 0, 
                   'fscore_weighted': 0}
    
    logger.info(f"\nTraining with seed {seed} just started...")

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        true_values, pred_values = [], []

        progress_bar = tqdm(enumerate(train_data), total=batches, desc=f"Epoch {epoch + 1}/{epochs}", leave=True)

        for _, (X, y, mask_X, mask_y) in progress_bar:
            X, y, mask_X, mask_y = X.to(device), y.long().to(device), mask_X.to(device), mask_y.to(device)
            y_pred = model(X, mask_X)

            batch_size, seq_len, _ = y_pred.size()
            y_pred = y_pred.reshape(batch_size * seq_len, num_classes)
            y = y.reshape(batch_size * seq_len)
            mask_y = mask_y.reshape(batch_size * seq_len)

            y_pred = utils.mask(tensor=y_pred, mask=mask_y, id=0)
            y = utils.mask(tensor=y, mask=mask_y, id=0)
            
            train_loss = criterion(pred=y_pred, true=y)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            total_train_loss += train_loss.item()
            progress_bar.set_postfix(Loss=train_loss.item())

            true_values.append(y.cpu().numpy())
            pred_values.append(y_pred.detach().cpu().numpy())

        avg_train_loss = total_train_loss / batches
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for X, y, mask_X, mask_y in val_data:
                X, y, mask_X, mask_y = X.to(device), y.long().to(device), mask_X.to(device), mask_y.to(device)
                y_pred = model(X, mask_X)

                batch_size, seq_len, _ = y_pred.size()
                y_pred = y_pred.reshape(batch_size * seq_len, num_classes)
                y = y.reshape(batch_size * seq_len)
                mask_y = mask_y.reshape(batch_size * seq_len)

                y_pred = utils.mask(tensor=y_pred, mask=mask_y, id=0)
                y = utils.mask(tensor=y, mask=mask_y, id=0)

                val_loss = criterion(pred=y_pred, true=y)
                total_val_loss += val_loss.item()
        
        avg_val_loss = total_val_loss / batches
        val_losses.append(avg_val_loss)

        true_values = np.concatenate(true_values)
        pred_values = np.concatenate(pred_values)

        true_classes = true_values.tolist()
        pred_classes = [utils.get_max(pred).index for pred in pred_values]
        
        logger.info(f"Epoch [{epoch + 1}/{epochs}], Training Loss: {avg_train_loss:.6f}, Validation Loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            stationary = 0

            logger.info(f"New best val found! ~ Epoch [{epoch + 1}/{epochs}], Val Loss {avg_val_loss}")

            mfn = utils.get_path(dirs=["models", "transformer", str(seed)], name="transformer.pth")
            torch.save(model.state_dict(), mfn)

            checkpoints.update({'best_epoch': epoch+1, 
                                'best_train_loss': avg_train_loss, 
                                'best_val_loss': best_val_loss, 
                                **utils.get_prfs(true=true_classes, pred=pred_classes)})

            if visualize:
                utils.visualize(type="heatmap",
                        values=(true_classes, pred_classes), 
                        labels=("True Values", "Predicted Values"), 
                        title="Train Heatmap "+ylabel,
                        classes=classes,
                        coloring=['gold', 'deepskyblue'],
                        path=utils.get_path(dirs=["models", "transformer", str(seed)]))
                
        else:
            stationary += 1

        if stationary >= patience:
            logger.info(f"Early stopping after {epoch + 1} epochs without improvement. Patience is {patience}.")
            break

        scheduler.step()

    cfn = utils.get_path(dirs=["models", "transformer", str(seed)], name="train_checkpoints.json")
    checkpoints.update({'epochs': epoch+1})
    utils.save_json(data=checkpoints, filename=cfn)
    
    if visualize:
        utils.visualize(type="multi-plot",
                        values=[(range(1, len(train_losses) + 1), train_losses), (range(1, len(val_losses) + 1), val_losses)], 
                        labels=("Epoch", "Loss"), 
                        title="Loss Curves",
                        plot_func=plt.plot,
                        coloring=['brown', 'royalblue'],
                        names=["Training", "Validation"],
                        path=utils.get_path(dirs=["models", "transformer", str(seed)]))

    logger.info(f'\nTraining with seed {seed} complete!\nFinal Training Loss: {avg_train_loss:.6f} & Validation Loss: {best_val_loss:.6f}\n')

    return avg_train_loss, best_val_loss

def main():
    path = "data/owm+plc/training_set_classif.csv"
    seq_len = 1440 // 180
    batch_size = 1
    classes = ["< 0.42 KWh", "< 1.05 KWh", "< 1.51 KWh", "< 2.14 KWh", ">= 2.14 KWh"]
    seeds = [6, 72, 157, 838, 1214, 1916]

    df = load(path=path, parse_dates=["DATETIME"], normalize=True)
    df_prep = prepare(df, phase="train")

    weights = utils.load_json(filename='static/weights.json')

    ds = TSDataset(df=df_prep, seq_len=seq_len, X=params["X"], t=params["t"], y=params["y"])
    ds_train, ds_val = split(ds, vperc=0.2)

    dl_train = DataLoader(ds_train, batch_size, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size, shuffle=False)

    for seed in seeds:
        model = Transformer(in_size=len(params["X"])+len(params["t"]), 
                            out_size=len(classes),
                            nhead=1, 
                            num_layers=1,
                            dim_feedforward=2048, 
                            dropout=0)

        _, _ = train(data=(dl_train, dl_val),
               classes=classes,
               epochs=300,
               patience=30,
               lr=5e-4,
               criterion=utils.WeightedCrossEntropyLoss(weights),
               model=model,
               optimizer="AdamW",
               scheduler=("StepLR", 1.0, 0.98),
               seed=seed,
               visualize=True)