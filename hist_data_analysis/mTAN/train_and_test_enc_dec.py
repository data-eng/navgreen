import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from .model import dec_mtan_rnn, enc_mtan_rnn, create_regressor
from .data_loader import load_df, TimeSeriesDataset
from .utils import compute_losses, mean_squared_error


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


torch.manual_seed(1505)
np.random.seed(1505)
torch.cuda.manual_seed(1505)

hp_cols = ["DATETIME", "BTES_TANK", "DHW_BUFFER", "POWER_HP", "Q_CON_HEAT"]
pvt_cols = ["DATETIME", "OUTDOOR_TEMP", "PYRANOMETER", "DHW_BOTTOM", "POWER_PVT", "Q_PVT"]



def evaluate_regressor(model, dataloader, device,  regressor, latent_dim, criterion, num_sample=1, plot=False, pred_values=None):
    model.eval()
    regressor.eval()

    true_values = []
    predicted_values = []

    total_loss = 0
    for (X, masks_X, observed_tp), y in dataloader:
        X, masks_X, y = X.to(device), masks_X.to(device), y.to(device)
        batch_len = X.shape[0]
        with torch.no_grad():
            out = model(X, observed_tp, masks_X)

            qz0_mean, qz0_logvar = out[:, :, :latent_dim], out[:, :, latent_dim:]
            epsilon = torch.randn(num_sample, qz0_mean.shape[0], qz0_mean.shape[1], qz0_mean.shape[2]).to(device)
            z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
            z0 = z0.view(-1, qz0_mean.shape[1], qz0_mean.shape[2])


            # y = y.unsqueeze(0).repeat_interleave(num_sample, 0).view(-1)
            y_ = y[:, -1, :]

            out = regressor(z0)
            out = out.view(batch_len, -1, y_.shape[1])[:, -1, :]

            total_loss += criterion(out, y_).item() * batch_len * num_sample
            true_values.append(y_.cpu().numpy())
            predicted_values.append(out.cpu().numpy())

    true_values = np.concatenate(true_values, axis=0)
    predicted_values = np.concatenate(predicted_values, axis=0)

    if plot:
        assert pred_values is not None

        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.scatter(true_values[:, 0], predicted_values[:, 0], color='lightblue')
        plt.xlabel("True Value")
        plt.ylabel("Predicted Value")
        plt.title(pred_values[0])

        plt.subplot(1, 2, 2)
        plt.scatter(true_values[:, 1], predicted_values[:, 1], color='lightcoral')
        plt.xlabel("True Value")
        plt.ylabel("Predicted Value")
        plt.title(pred_values[1])

        plt.tight_layout()
        plt.savefig(f"{pred_values[0]}_{pred_values[1]}_dec.png", dpi=300)

    #return total_loss / len(dataloader)
    return total_loss/predicted_values.shape[0]


def train(enc, dec, regr, train_loader, val_loader, checkpoint_pth, criterion, task, learning_rate, epochs, patience,
          k_iwae, latent_dim, alpha=1., kl=True):

    # Early stopping variables
    best_val_loss = float('inf')
    final_train_recon_loss = float('inf')
    final_train_ce_loss = float('inf')
    final_mse = float('inf')
    epochs_without_improvement = 0


    #params = (list(enc.parameters()) + list(dec.parameters()) + list(regr.parameters()))

    params = list(enc.parameters()) + list(dec.parameters()) + list(regr.parameters())

    optimizer = optim.Adam(params, lr=learning_rate)

    ii = 0

    if checkpoint_pth is not None:
        checkpoint = torch.load(checkpoint_pth)
        enc.load_state_dict(checkpoint['enc_state_dict'])
        dec.load_state_dict(checkpoint['dec_state_dict'])
        regr.load_state_dict(checkpoint['regressor_dict']),
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('loading saved weights', checkpoint['epoch'])

    for epoch in range(1, epochs + 1):
        enc.train()
        dec.train()
        regr.train()

        start_time = time.time()

        train_recon_loss, train_ce_loss = 0, 0
        mse = 0
        if kl:
            wait_until_kl_inc = 10
            if epoch < wait_until_kl_inc:
                kl_coef = 0.
            else:
                kl_coef = (1 - 0.99 ** (epoch - wait_until_kl_inc))
        else:
            kl_coef = 1

        for (X, masks_X, observed_tp), y in train_loader:
            X, masks_X, y = X.to(device), masks_X.to(device), y.to(device)

            batch_len = X.shape[0]

            out = enc(X, observed_tp, masks_X)
            # print(f'out: {out}')
            qz0_mean, qz0_logvar = out[:, :, :latent_dim], out[:, :, latent_dim:]
            epsilon = torch.randn(k_iwae, qz0_mean.shape[0], qz0_mean.shape[1], qz0_mean.shape[2]).to(device)
            z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
            print(f"out {out}")
            z0 = z0.view(-1, qz0_mean.shape[1], qz0_mean.shape[2])
            pred_y = regr(z0)
            #print(f"pred_y {pred_y}")
            pred_x = dec(z0, observed_tp[None, :, :].repeat(k_iwae, 1, 1).view(-1, observed_tp.shape[1]))
            pred_x = pred_x.view(k_iwae, batch_len, pred_x.shape[1], pred_x.shape[2])  # nsample, batch, seqlen, dim
            # compute loss
            logpx, analytic_kl = compute_losses(qz0_mean, qz0_logvar, pred_x, device, X, masks_X)
            #print(f"pred_x {pred_x}")
            ii += 1
            if ii == 2:
                break
            #print(f"logpx = {logpx} and analytic_kl = {analytic_kl}")
            recon_loss = -(torch.logsumexp(logpx - kl_coef * analytic_kl, dim=0).mean(0) - np.log(k_iwae))
            #y = y.unsqueeze(0).repeat_interleave(k_iwae, 0).view(-1)
            y_ = y[:, -1, :]
            pred_y = pred_y.view(batch_len, -1, y_.shape[1])[:, -1, :]
            ce_loss = criterion(pred_y, y_)
            #print(f"ce_loss = {ce_loss:.5f} and recon_loss = {recon_loss:.5f}")
            loss = recon_loss + alpha * ce_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #print(f'ce_loss: {ce_loss}, pred_y {pred_y}, y_ {y_}')
            train_ce_loss += ce_loss.item() #* batch_len
            train_recon_loss += recon_loss.item() #* batch_len
            mse += mean_squared_error(X, pred_x.mean(0), masks_X) #* batch_len

        val_loss = evaluate_regressor(model=enc, dataloader=val_loader, latent_dim=latent_dim, regressor=regr,
                                      num_sample=1, device=device, criterion=criterion)

        train_recon_loss_ = train_recon_loss / len(train_loader)
        train_ce_loss_ = train_ce_loss / len(train_loader)
        mse_ = mse / len(train_loader)

        print(f'Epoch {epoch} | train_recon_loss: {train_recon_loss_:.6f}, train_ce_loss: {train_ce_loss_:.6f}, '
              f'mse_: {mse_:.6f}, '#val_loss: {val_loss:.6f}, '
              f'Time : {(time.time() - start_time) / 60:.2f} minutes')

        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            final_train_recon_loss = train_recon_loss_
            final_train_ce_loss = train_ce_loss_
            final_mse = mse_
            epochs_without_improvement = 0

            torch.save({
                'epoch': epoch,
                'enc_state_dict': enc.state_dict(),
                'dec_state_dict': dec.state_dict(),
                'regressor_dict' : regr.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'final_train_recon_loss': final_train_recon_loss,
                'final_train_ce_loss': final_train_ce_loss,
                'final_mse': final_mse,
            }, f'best_model_{task}_dec.pth')
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping after {epoch} epochs without improvement. Patience is {patience}.")
            break

    print("Training complete!")

    return final_train_recon_loss, final_train_ce_loss, best_val_loss


def main_loop():

    validation_set_percentage = 0.2

    print("TASK 1 | Train and evaluate on HP related prediction")
    task = "hp"

    with open("hist_data_analysis/mTAN/best_model_params_hp.json", 'r') as file:
        params = json.load(file)

    dim = 2
    # Parameters:
    num_heads = params["num_heads"]
    # rec_hidden = params["rec_hidden"]
    embed_time = params["embed_time"]

    sequence_length = params["sequence_length"]
    batch_size = params["batch_size"]
    grp = params["grp"]

    #num_ref_points = 32
    latent_dim = 16
    rec_hidden = 32
    dec_hidden = 32
    k_iwae = 1

    lr = params["lr"]
    epochs = 1
    patience = 8

    #print(f"Parameters : num_heads = {num_heads}, rec_hidden = {rec_hidden}, embed_time = {embed_time}, "
    #      f"sequence_length = {sequence_length}, batch_size = {batch_size}, grp = {grp}")

    df_path_train = "data/training_set_before_conv.csv"
    (_, df_hp_train, _), mean_stds = load_df(df_path=df_path_train, hp_cols=hp_cols, pvt_cols=pvt_cols,
                                             parse_dates=["Date&time"], normalize=True, grp=grp, hist_data=True)

    df_path_test = "data/test_set_before_conv.csv"
    (_, df_hp_test, _), _ = load_df(df_path=df_path_test, hp_cols=hp_cols, pvt_cols=pvt_cols, parse_dates=["Date&time"],
                                    normalize=True, grp=grp, hist_data=True, stats=mean_stds)

    df_hp_train.to_csv("hp_df_train.csv")
    df_hp_test.to_csv("hp_df_test.csv")

    train_df = pd.read_csv("hp_df_train.csv", parse_dates=['DATETIME'], index_col='DATETIME')
    test_df = pd.read_csv("hp_df_test.csv", parse_dates=['DATETIME'], index_col='DATETIME')

    X_cols = ["BTES_TANK", "DHW_BUFFER"]
    y_cols = ["POWER_HP", "Q_CON_HEAT"]

    # Create a dataset and dataloader
    training_dataset = TimeSeriesDataset(dataframe=train_df, sequence_length=sequence_length,
                                         X_cols=X_cols, y_cols=y_cols, final_train=True)

    # Get the total number of samples and compute size of each corresponding set
    total_samples = len(training_dataset)
    validation_size = int(validation_set_percentage * total_samples)
    train_size = total_samples - validation_size

    # Use random_split to create training and validation datasets
    train_dataset, val_dataset = random_split(training_dataset, [train_size, validation_size])

    # Create dataloaders for training and validation sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print("Train dataloader loaded")
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print("Validation dataloader loaded")

    # Configure models
    enc = enc_mtan_rnn(input_dim=dim, query=torch.linspace(0, 1., embed_time), latent_dim=latent_dim, nhidden=rec_hidden,
                              embed_time=embed_time, num_heads=num_heads, device=device).to(device)

    dec = dec_mtan_rnn(input_dim=dim, query=torch.linspace(0, 1., embed_time), latent_dim=latent_dim, nhidden=dec_hidden,
        embed_time=embed_time, num_heads=num_heads, device=device).to(device)

    regressor = create_regressor(latent_dim, rec_hidden).to(device)

    # MSE loss
    criterion = nn.MSELoss()
    # Train the model
    training_rec_loss, training_classif_loss, validation_loss =  train(enc=enc, dec=dec, regr=regressor,
                                                                       train_loader=train_loader, val_loader=val_loader,
                                                                       checkpoint_pth=None, criterion=criterion, task=task,
                                                                       learning_rate=lr, epochs=epochs,  patience=patience,
                                                                       k_iwae=k_iwae, latent_dim=latent_dim)

    print(f'Final reconstruction training Loss : {training_rec_loss:.6f} & '
          f'classification training Loss : {training_classif_loss:.6f} &  '
          f'Validation Loss : {validation_loss:.6f}\n')

    # Create a dataset and dataloader
    testing_dataset = TimeSeriesDataset(dataframe=test_df, sequence_length=sequence_length,
                                         X_cols=y_cols, y_cols=y_cols)

    test_loader = DataLoader(testing_dataset, batch_size=batch_size, shuffle=False)
    print("Test dataloader loaded")

    # Configure models
    enc_trained_model = enc_mtan_rnn(input_dim=dim, query=torch.linspace(0, 1., embed_time), latent_dim=latent_dim,
                       nhidden=rec_hidden,
                       embed_time=embed_time, num_heads=num_heads, device=device).to(device)

    regressor_trained_model = create_regressor(latent_dim, rec_hidden).to(device)

    checkpoint = torch.load(f'best_model_{task}_dec.pth')
    enc_trained_model.load_state_dict(checkpoint['enc_state_dict'])
    regressor_trained_model.load_state_dict(checkpoint['regressor_dict'])

    # Test model's performance on unseen data
    testing_loss = evaluate_regressor(model=enc_trained_model, dataloader=val_loader, latent_dim=latent_dim,
                                      regressor=regressor_trained_model, num_sample=1, device=device,
                                      criterion=criterion, plot=True, pred_values=y_cols)

    print(f'Testing Loss (MSE) : {testing_loss:.6f}')


    print("\nTASK 2 | Train and evaluate on PVT related prediction")
    task = "pvt"

    with open("hist_data_analysis/mTAN/best_model_params_pvt.json", 'r') as file:
        params = json.load(file)

    dim = 3
    # Parameters:
    num_heads = params["num_heads"]
    rec_hidden = params["rec_hidden"]
    embed_time = params["embed_time"]
    sequence_length = params["sequence_length"]
    batch_size = params["batch_size"]
    lr = params["lr"]
    grp = params["grp"]

    print(f"Parameters : num_heads = {num_heads}, rec_hidden = {rec_hidden}, embed_time = {embed_time}, "
          f"sequence_length = {sequence_length}, batch_size = {batch_size}, grp = {grp}")

    df_path_train = "data/training_set_before_conv.csv"
    (_, _, df_pvt_train), mean_stds = load_df(df_path=df_path_train, hp_cols=hp_cols, pvt_cols=pvt_cols,
                                              parse_dates=["Date&time"], normalize=True, grp=grp, hist_data=True)

    df_path_test = "data/test_set_before_conv.csv"
    (_, _, df_pvt_test), _ = load_df(df_path=df_path_test, hp_cols=hp_cols, pvt_cols=pvt_cols, parse_dates=["Date&time"],
                                     normalize=True, grp=grp, hist_data=True, stats=mean_stds)

    df_pvt_train.to_csv("pvt_df_train.csv")
    df_pvt_test.to_csv("pvt_df_test.csv")

    train_df = pd.read_csv("pvt_df_train.csv", parse_dates=['DATETIME'], index_col='DATETIME')
    test_df = pd.read_csv("pvt_df_test.csv", parse_dates=['DATETIME'], index_col='DATETIME')

    X_cols = ["OUTDOOR_TEMP", "PYRANOMETER", "DHW_BOTTOM"]
    y_cols = ["POWER_PVT", "Q_PVT"]

    # Create a dataset and dataloader
    training_dataset = TimeSeriesDataset(dataframe=train_df, sequence_length=sequence_length,
                                         X_cols=X_cols, y_cols=y_cols, final_train=True)

    # Get the total number of samples and compute size of each corresponding set
    total_samples = len(training_dataset)
    validation_size = int(validation_set_percentage * total_samples)
    train_size = total_samples - validation_size

    # Use random_split to create training and validation datasets
    train_dataset, val_dataset = random_split(training_dataset, [train_size, validation_size])

    # Create dataloaders for training and validation sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print("Train dataloader loaded")
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print("Validation dataloader loaded")

    # Configure models
    enc = enc_mtan_rnn(input_dim=dim, query=torch.linspace(0, 1., embed_time), latent_dim=latent_dim, nhidden=rec_hidden,
                              embed_time=embed_time, num_heads=num_heads, device=device).to(device)

    dec = dec_mtan_rnn(input_dim=dim, query=torch.linspace(0, 1., embed_time), latent_dim=latent_dim, nhidden=dec_hidden,
        embed_time=embed_time, num_heads=num_heads, device=device).to(device)

    regressor = create_regressor(latent_dim, rec_hidden).to(device)

    # MSE loss
    criterion = nn.MSELoss()
    # Train the model
    training_rec_loss, training_classif_loss, validation_loss =  train(enc=enc, dec=dec, regr=regressor,
                                                                       train_loader=train_loader, val_loader=val_loader,
                                                                       checkpoint_pth=None, criterion=criterion, task=task,
                                                                       learning_rate=lr, epochs=epochs,  patience=patience,
                                                                       k_iwae=k_iwae, latent_dim=latent_dim)

    print(f'Final reconstruction training Loss : {training_rec_loss:.6f} & '
          f'classification training Loss : {training_classif_loss:.6f} &  '
          f'Validation Loss : {validation_loss:.6f}\n')

    # Create a dataset and dataloader
    testing_dataset = TimeSeriesDataset(dataframe=test_df, sequence_length=sequence_length,
                                         X_cols=y_cols, y_cols=y_cols)

    test_loader = DataLoader(testing_dataset, batch_size=batch_size, shuffle=False)
    print("Test dataloader loaded")

    # Configure models
    enc_trained_model = enc_mtan_rnn(input_dim=dim, query=torch.linspace(0, 1., embed_time), latent_dim=latent_dim,
                       nhidden=rec_hidden,
                       embed_time=embed_time, num_heads=num_heads, device=device).to(device)

    regressor_trained_model = create_regressor(latent_dim, rec_hidden).to(device)

    checkpoint = torch.load(f'best_model_{task}_dec.pth')
    enc_trained_model.load_state_dict(checkpoint['enc_state_dict'])
    regressor_trained_model.load_state_dict(checkpoint['regressor_dict'])

    # Test model's performance on unseen data
    testing_loss = evaluate_regressor(model=enc_trained_model, dataloader=val_loader, latent_dim=latent_dim,
                                      regressor=regressor_trained_model, num_sample=1, device=device,
                                      criterion=criterion, plot=True, pred_values=y_cols)

    print(f'Testing Loss (MSE) : {testing_loss:.6f}')


