import torch

EPSILON = 1e-12

def mine_loss(data_batch,classifier):
    join_sample = data_batch["join"]
    independent_sample = data_batch['independent']

    T = classifier(join_sample)
    iT = classifier(independent_sample)

    # Apply log-sum-exp trick for stability
    max_T = torch.max(iT)
    log_mean_exp_T = max_T + torch.log(torch.mean(torch.exp(iT - max_T)))

    mine = T.mean() - log_mean_exp_T
    mine = -mine

    return mine