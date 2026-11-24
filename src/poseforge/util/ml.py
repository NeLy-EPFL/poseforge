import torch


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    """Resolve a device specifier to a concrete ``torch.device``.

    ``"auto"`` prefers a discrete GPU (CUDA), then Apple Silicon (MPS), then
    falls back to CPU, so the same training/inference code runs fast on a Linux
    CUDA box and on a Mac without any change. Any explicit value (``"cpu"``,
    ``"cuda"``, ``"mps"`` or a ``torch.device``) is passed through unchanged.
    """
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


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
