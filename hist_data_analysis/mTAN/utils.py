import torch
import numpy as np


def log_normal_pdf(x, mean, logvar, mask):
    const = torch.from_numpy(np.array([2. * np.pi])).float().to(x.device)
    const = torch.log(const)
    # print(f"(torch.exp(logvar)) = {torch.exp(logvar)}")
    #print(f"-.5 * (const + logvar + (x - mean) ** 2. / torch.exp(logvar)) = {-.5 * (const + logvar + (x - mean) ** 2. / torch.exp(logvar))}")
    return -.5 * (const + logvar + (x - mean) ** 2. / torch.exp(logvar)) * mask


def normal_kl(mu1, lv1, mu2, lv2):
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    lstd1 = lv1 / 2.
    lstd2 = lv2 / 2.

    kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - .5
    return kl


def mean_squared_error(orig, pred, mask):
    error = (orig - pred) ** 2
    error = error * mask
    return error.sum() / mask.sum()


def compute_losses(qz0_mean, qz0_logvar, pred_x, device, observed_data, observed_mask):
    # noise_std = args.std  # default 0.1
    noise_std = 0.1
    noise_std_ = torch.zeros(pred_x.size()).to(device) + noise_std
    noise_logvar = 2. * torch.log(noise_std_).to(device)
    logpx = log_normal_pdf(observed_data, pred_x, noise_logvar,
                           observed_mask).sum(-1).sum(-1)

    pz0_mean = pz0_logvar = torch.zeros(qz0_mean.size()).to(device)
    analytic_kl = normal_kl(qz0_mean, qz0_logvar,
                            pz0_mean, pz0_logvar).sum(-1).sum(-1)

    # Normalize for a smaller reconstruction error
    logpx /= observed_mask.sum(-1).sum(-1)
    analytic_kl /= observed_mask.sum(-1).sum(-1)

    # Replace the nans and infs coming from zero division
    logpx = torch.nan_to_num(logpx, posinf=0., neginf=0.,nan=0.)
    analytic_kl = torch.nan_to_num(analytic_kl, posinf=0., neginf=0.,nan=0.)

    return logpx, analytic_kl
