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

def contrastive_loss_multilayer(features_gat1: torch.Tensor, features_gat2: torch.Tensor, 
                                features_transformer1: torch.Tensor, features_transformer2: torch.Tensor,
                                first_mlp_features1: torch.Tensor, first_mlp_features2: torch.Tensor,
                                is_same_smiles: torch.Tensor,
                                margin: float = 5.0, weights = (0.4, 0.3, 0.3), mask= None,) -> torch.Tensor:
    """
    Compute the contrastive loss across multiple layers.

    :param features_gat1, features_gat2: Features from GAT encoding for the two inputs.
    :param features_transformer1, features_transformer2: Features from Transformer layer for the two inputs.
    :param is_same_smiles: A tensor indicating if the two inputs have the same SMILES.
    :param margin: The margin for contrastive loss.
    :param weights: The weights for combining the contrastive losses from each feature layer.
    :return: The combined contrastive loss.
    """
    
    # Helper function to compute the contrastive loss for a pair of features
    def single_contrastive_loss(features1: torch.Tensor, features2: torch.Tensor, is_same_smiles: torch.Tensor, margin: float, mask = None) -> torch.Tensor:
        
        euclidean_distance = torch.norm(features1 - features2, dim=-1)
        # euclidean_distance = (features1 - features2).pow(2).sum(1)
        if mask is None:
            mask = torch.ones_like(euclidean_distance)  # Assuming features are of shape [batch_size, seq_len, feature_dim]
        
        # Apply the mask to the euclidean_distance
        euclidean_distance = euclidean_distance * mask    
        # Convert is_same_smiles to float for computation
        is_same_smiles_float = is_same_smiles.float().unsqueeze(1)

        # Compute the loss for same and different smiles
        same_smiles_loss = euclidean_distance
        different_smiles_loss = torch.clamp(margin - euclidean_distance, min=0.0)

        # Combine the two losses based on the is_same_smiles tensor
        combined_loss = is_same_smiles_float * same_smiles_loss + (1 - is_same_smiles_float) * different_smiles_loss

        return combined_loss

    # Compute the contrastive losses for each feature layer
    loss_gat = single_contrastive_loss(features_gat1, features_gat2, is_same_smiles, margin, mask)
    loss_transformer = single_contrastive_loss(features_transformer1, features_transformer2, is_same_smiles, margin, mask)
    loss_mlp = single_contrastive_loss(first_mlp_features1, first_mlp_features2, is_same_smiles, margin, mask)

    # Combine the contrastive losses with the given weights
    mean_loss = weights[0] * loss_gat + weights[1] * loss_transformer + weights[2] * loss_mlp

    return mean_loss