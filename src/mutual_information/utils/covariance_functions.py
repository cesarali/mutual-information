import torch

def copy_upper_diagonal_values(tensor):
    # Create a sample 2D tensor (replace this with your own tensor)
    original_tensor = tensor
    # Get the upper triangular portion of the tensor (above the diagonal)
    upper_triangular = torch.triu(original_tensor, diagonal=1)
    # Create a mask for the diagonal and below
    mask = torch.tril(torch.ones_like(original_tensor), diagonal=0)
    # Fill the lower triangular portion with values from the upper triangular portion
    result_tensor = (upper_triangular + upper_triangular.transpose(0, 1))
    # Add the diagonal elements from the original tensor
    result_tensor += torch.diag_embed(torch.diag(original_tensor))
    return result_tensor