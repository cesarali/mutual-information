import torch

EPSILON = 1e-12

def contrastive_loss(data_batch,binary_classifier):
    join_sample = data_batch["join"]
    independent_sample = data_batch['independent']

    join_probability_estimate = binary_classifier(join_sample)
    independent_probability_estimate = binary_classifier(independent_sample)
    loss_1 = torch.log(join_probability_estimate + EPSILON)
    loss_2 = torch.log(1.- independent_probability_estimate + EPSILON)
    loss = loss_1.mean() + loss_2.mean()
    return -loss