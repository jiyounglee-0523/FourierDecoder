import torch

import numpy as np

def kl_divergence(mu_1, sigma_1, mu_2, sigma_2):

    logvar_q = sigma_1.square().log()
    logvar_p = sigma_2.square().log()

    mu_q = mu_1
    mu_p = mu_2

    var_p = torch.exp(logvar_p)
    kl_div = (torch.exp(logvar_q) + (mu_q - mu_p) ** 2) / var_p \
             - 1.0 \
             + logvar_p - logvar_q
    kl_div = 0.5 * kl_div.sum(1).mean()

    return kl_div

def log_normal_pdf(x, mean, logvar):
    const = torch.from_numpy(np.array([2. * np.pi])).float().to(x.device)
    const = torch.log(const)
    return -.5 * (const + logvar + (x-mean)**2. / torch.exp(logvar))