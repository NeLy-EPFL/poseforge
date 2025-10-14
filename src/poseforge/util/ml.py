import torch


def count_optimizer_parameters(
    optimizer: torch.optim.Optimizer, trainable_only: bool = True
) -> int:
    """Counts the number of parameters handled by a PyTorch optimizer,
    optionally limited to trainable parameters (requires_grad==True)."""
    total_params = 0
    for group in optimizer.param_groups:
        for p in group["params"]:
            if not trainable_only or p.requires_grad:
                total_params += p.numel()
    return total_params


def count_module_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """
    Counts the total number of parameters in a PyTorch nn.Module,
    optionally limited to trainable parameters (requires_grad==True)."""
    total_params = 0
    for p in model.parameters():
        if not trainable_only or p.requires_grad:
            total_params += p.numel()
    return total_params
