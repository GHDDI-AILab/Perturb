"""Code from https://github.com/bowang-lab/scGPT/blob/main/scgpt/loss.py."""

import torch
import torch.nn.functional as F


def masked_mse_loss(
    input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute the masked MSE loss between input and target.
    """
    mask = mask.float()
    loss = F.mse_loss(input * mask, target * mask, reduction="sum")
    return loss / mask.sum()


def criterion_neg_log_bernoulli(
    input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute the negative log-likelihood of Bernoulli distribution
    """
    mask = mask.float()
    bernoulli = torch.distributions.Bernoulli(probs=input)
    masked_log_probs = bernoulli.log_prob((target > 0).float()) * mask
    return -masked_log_probs.sum() / mask.sum()


def masked_relative_error(
    input: torch.Tensor, target: torch.Tensor, mask: torch.LongTensor
) -> torch.Tensor:
    """
    Compute the masked relative error between input and target.
    """
    assert mask.any()
    loss = torch.abs(input[mask] - target[mask]) / (target[mask] + 1e-6)
    return loss.mean()

def weighted_mse_loss(control: torch.Tensor, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Custom weighted MSE loss.

    Args:
    - control (torch.Tensor): Control tensor.
    - target (torch.Tensor): Perturbed tensor.
    - output (torch.Tensor): Model Output tensor.

    Returns:
    - torch.Tensor: The computed loss value.
    """
    # Convert tensors to float if necessary
    if control.dtype != torch.float32:
        control = control.float()
    if output.dtype != torch.float32:
        output = output.float()
    if target.dtype != torch.float32:
        target = target.float()
    
    mse_loss = F.mse_loss(output, target, reduction="mean")
    weights = torch.sigmoid(torch.abs(target - control))
    weighted_mse = weights * (target - output)**2
    loss = torch.mean(weighted_mse) + mse_loss
    
    return loss

def pearson_corr_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute the Pearson correlation coefficient loss.

    Args:
    output (torch.Tensor): The model output. Shape: (n_samples,) or (n_samples, n_outputs).
    target (torch.Tensor): The target (true) values. Shape should be the same as output.

    Returns:
    torch.Tensor: A tensor containing a single scalar value, the loss.

    Note:
    The output and target should be one-dimensional or two-dimensional tensors of the same shape and data type.
    This loss function does not support broadcasting.
    """
    if output.shape != target.shape:
        raise ValueError("Output and target should have the same shape, "
                         f"got output shape {output.shape} and target shape {target.shape}")

    # Convert tensors to float if necessary
    if output.dtype != torch.float32:
        output = output.float()
    if target.dtype != torch.float32:
        target = target.float()

    # Compute loss for each sample in the batch
    mean_output = torch.mean(output, dim=1, keepdim=True)
    mean_target = torch.mean(target, dim=1, keepdim=True)
    std_output = torch.std(output, dim=1, keepdim=True)
    std_target = torch.std(target, dim=1, keepdim=True)
    pearson_corr = torch.mean((output - mean_output) * (target - mean_target), dim=1) / (std_output * std_target)

    # Average the losses across the batch
    return torch.mean(1 - pearson_corr)

def delta_pearson_corr_loss(control: torch.Tensor, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute the Pearson correlation coefficient loss.

    Args:
    control (torch.Tensor): The control (true) values. Shape: (n_samples,) or (n_samples, n_outputs).
    output (torch.Tensor): The model output. Shape: (n_samples,) or (n_samples, n_outputs).
    target (torch.Tensor): The target (true) values. Shape should be the same as output.

    Returns:
    torch.Tensor: A tensor containing a single scalar value, the loss.

    Note:
    The output and target should be one-dimensional or two-dimensional tensors of the same shape and data type.
    This loss function does not support broadcasting.
    """
    if output.shape != target.shape:
        raise ValueError("Output and target should have the same shape, "
                         f"got output shape {output.shape} and target shape {target.shape}")

    # Convert tensors to float if necessary
    if control.dtype != torch.float32:
        control = control.float()
    if output.dtype != torch.float32:
        output = output.float()
    if target.dtype != torch.float32:
        target = target.float()
    
    delta_output = output - control
    delta_target = target - control

    # Compute loss for each sample in the batch
    mean_output = torch.mean(delta_output, dim=1, keepdim=True)
    mean_target = torch.mean(delta_target, dim=1, keepdim=True)
    std_output = torch.std(delta_output, dim=1, keepdim=True)
    std_target = torch.std(delta_target, dim=1, keepdim=True)
    pearson_corr = torch.mean((delta_output - mean_output) * (delta_target - mean_target), dim=1) / (std_output * std_target)

    # Average the losses across the batch
    return torch.mean(1 - pearson_corr)

def delta_cosine_loss(control: torch.Tensor, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute the Pearson correlation coefficient loss.

    Args:
    control (torch.Tensor): The control (true) values. Shape: (n_samples,) or (n_samples, n_outputs).
    output (torch.Tensor): The model output. Shape: (n_samples,) or (n_samples, n_outputs).
    target (torch.Tensor): The target (true) values. Shape should be the same as output.

    Returns:
    torch.Tensor: A tensor containing a single scalar value, the loss.

    Note:
    The output and target should be one-dimensional or two-dimensional tensors of the same shape and data type.
    This loss function does not support broadcasting.
    """
    if output.shape != target.shape:
        raise ValueError("Output and target should have the same shape, "
                         f"got output shape {output.shape} and target shape {target.shape}")

    # Convert tensors to float if necessary
    if control.dtype != torch.float32:
        control = control.float()
    if output.dtype != torch.float32:
        output = output.float()
    if target.dtype != torch.float32:
        target = target.float()
    
    delta_output = output - control
    delta_target = target - control

    delta_output = output
    delta_target = target

    # Compute loss for each sample in the batch
    cos_sim = F.cosine_similarity(delta_output, delta_target, dim=1)
    cos_loss = 1.0 - cos_sim.mean()

    # Average the losses across the batch
    return cos_loss
