from torch.utils.data import DataLoader, ConcatDataset
import matplotlib.pyplot as plt
import utils

from .model import Transformer
from .loader import *

from .loader_init import load as load_init
from .loader_init import prepare as prepare_init

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s:%(lineno)d:%(levelname)s:%(name)s:%(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f'Device is {device}')

# Previous Xs, check sorting
#
# New Xs, check sorting
# "X": ["TEMPERATURE", "HUMIDITY", "WIND_SPEED", "WIND_DIRECTION", "SKY"],

params_init = {"X": ["humidity", "pressure", "feels_like", "temp", "wind_speed", "rain_1h"]}


def train(data, data_combined, classes, epochs, patience, lr, criterion, model, optimizer, scheduler, seed, y,
          whole_dataset_epoch, visualize=False, dir_name="tuned"):
    model.to(device)
    train_data_new, val_data_new = data
    train_data_combined, val_data_combined = data_combined
    num_classes = len(classes)
    optimizer = utils.get_optim(optimizer, model, lr)
    scheduler = utils.get_sched(*scheduler, optimizer)

    best_val_loss = float('inf')
    ylabel = y[0]
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

    train_data = train_data_new
    val_data = val_data_new

    for epoch in range(epochs):

        # At some point continue training with the whole dataset to avoid catastrophic forgetting
        if epoch == whole_dataset_epoch:
            train_data = train_data_combined
            val_data = val_data_combined

        model.train()
        total_train_loss = 0.0
        true_values, pred_values = [], []

        for _, (X, y, mask_X, mask_y) in enumerate(train_data):
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

            true_values.append(y.cpu().numpy())
            pred_values.append(y_pred.detach().cpu().numpy())

        avg_train_loss = total_train_loss / len(train_data)
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

        avg_val_loss = total_val_loss / len(val_data)
        val_losses.append(avg_val_loss)

        true_values = np.concatenate(true_values)
        pred_values = np.concatenate(pred_values)

        true_classes = true_values.tolist()
        pred_classes = [utils.get_max(pred).index for pred in pred_values]

        logger.info(f"Epoch [{epoch + 1}/{epochs}], Training Loss: {avg_train_loss:.6f}, "
                    f"Validation Loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            stationary = 0

            logger.info(f"New best val found! ~ Epoch [{epoch + 1}/{epochs}], Val Loss {avg_val_loss}")

            mfn = utils.get_path(dirs=["models", ylabel, "transformer", dir_name, str(seed)], name="transformer.pth")
            torch.save(model.state_dict(), mfn)

            checkpoints.update({'best_epoch': epoch + 1,
                                'best_train_loss': avg_train_loss,
                                'best_val_loss': best_val_loss,
                                **utils.get_prfs(true=true_classes, pred=pred_classes)})

            if visualize:
                utils.visualize(type="heatmap",
                                values=(true_classes, pred_classes),
                                labels=("True Values", "Predicted Values"),
                                title="Train Heatmap " + ylabel,
                                classes=classes,
                                coloring=['azure', 'darkblue'],
                                path=utils.get_path(dirs=["models", ylabel, "transformer", dir_name, str(seed)]))

        else:
            stationary += 1

        if stationary >= patience:
            logger.info(f"Early stopping after {epoch + 1} epochs without improvement. Patience is {patience}.")
            break

        scheduler.step()

    cfn = utils.get_path(dirs=["models", ylabel, "transformer", dir_name, str(seed)], name="train_checkpoints.json")
    checkpoints.update({'epochs': epoch + 1})
    utils.save_json(data=checkpoints, filename=cfn)

    if visualize:
        cfn = utils.get_path(dirs=["models", ylabel, "transformer", dir_name, str(seed)], name="train_losses.json")
        utils.save_json(data=train_losses, filename=cfn)

        utils.visualize(type="multi-plot",
                        values=[(range(1, len(train_losses) + 1), train_losses),
                                (range(1, len(val_losses) + 1), val_losses)],
                        labels=("Epoch", "Loss"),
                        title="Loss Curves",
                        plot_func=plt.plot,
                        coloring=['brown', 'royalblue'],
                        names=["Training", "Validation"],
                        path=utils.get_path(dirs=["models", ylabel, "transformer", dir_name, str(seed)]))

    logger.info(f'\nTraining with seed {seed} complete!\nFinal Training Loss: {avg_train_loss:.6f} '
                f'& Validation Loss: {best_val_loss:.6f}\n')

    return avg_train_loss, best_val_loss


def main_loop(seed, y_col):
    utils.set_seed(seed)

    path = "../../../data/train_classif_meteo.csv"
    path_init = "../../../data/training_set_classif_new_classes.csv"

    seq_len = 24 // 3
    batch_size = 8
    classes = ["0", "1", "2", "3", "4"]

    df_init = load_init(path=path_init, parse_dates=["DATETIME"], normalize=True, bin=y_col[0])
    df = load(path=path, parse_dates=["DATETIME"], normalize=True, bin=y_col[0])
    df_prep_init = prepare_init(df_init, phase="train")
    df_prep = prepare(df, phase="train")

    # Weights of outcome classes come from the initial data!
    weights = utils.load_json(filename=f'transformer/weights_{y_col[0]}.json')

    ds = TSDataset(df=df_prep, seq_len=seq_len, X=params["X"], t=params["t"], y=y_col, tune=True)
    ds_init = TSDataset(df=df_prep_init, seq_len=seq_len, X=params_init["X"], t=params["t"], y=y_col, tune=False)

    ds_combined = ConcatDataset([ds, ds_init])

    ds_train, ds_val = split(ds, vperc=0.2)
    ds_train_combined, ds_val_combined = split(ds_combined, vperc=0.2)

    dl_train = DataLoader(ds_train, batch_size, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size, shuffle=False)

    dl_train_combined = DataLoader(ds_train_combined, batch_size, shuffle=True)
    dl_val_combined = DataLoader(ds_val_combined, batch_size, shuffle=False)

    model = Transformer(in_size=len(params_init["X"])+len(params["t"]),
                        out_size=len(classes),
                        nhead=1,
                        num_layers=1,
                        dim_feedforward=2048,
                        dropout=0)

    trained_model_pth = utils.get_path(dirs=["..",
                                             "..",
                                             "public_weather",
                                             "hist_data_analysis",
                                             "models",
                                             y_col[0],
                                             "transformer",
                                             str(seed)],
                                       name="transformer.pth")

    model.load_state_dict(torch.load(trained_model_pth))

    epoch_num = 600
    whole_dataset_epoch = epoch_num//2
    dir_name = "tuned_whole" if whole_dataset_epoch < epoch_num else "tuned_new"

    _, _ = train(data=(dl_train, dl_val),
                 data_combined=(dl_train_combined, dl_val_combined),
                 classes=classes,
                 epochs=epoch_num,
                 patience=30,
                 lr=1e-5,
                 criterion=utils.WeightedCrossEntropyLoss(weights),
                 model=model,
                 optimizer="AdamW",
                 scheduler=("StepLR", 1.0, 0.98),
                 seed=seed,
                 y=y_col,
                 visualize=True,
                 whole_dataset_epoch=whole_dataset_epoch,
                 dir_name=dir_name)
