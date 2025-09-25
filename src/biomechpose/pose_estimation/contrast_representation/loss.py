import torch
import torch.nn as nn
import torch.nn.functional as F


def info_nce_loss(
    embeddings: torch.Tensor,
    temperature: float,
    n_samples: int,
    n_variants: int,
    device: torch.device | str,
) -> float:
    """Compute the InfoNCE loss, treating the same frame from different
    variants as positive pairs and different frames as negative pairs.

    Args:
        embeddings (torch.Tensor): Feature matrix of shape
            (batch_size * n_variants, feature_dim) in the embedding
            space wherein mutual information is evaluated.
        temperature (float): Temperature parameter for scaling the
            logits.
        batch_size (int): Number of unique frames in the batch.
        n_variants (int): Number of variants (e.g., different
            augmentations) per unique frame.
        device (torch.device | str): Device to perform computations on.

    Returns:
        torch.Tensor: InfoNCE loss as a single float value.
    """
    # Construct labels for binary classification
    frame_id = torch.cat([torch.arange(n_samples) for i in range(n_variants)], dim=0)
    labels_matrix = (frame_id[None, :] == frame_id[:, None]).to(device)
    # can be commented out after testing
    assert labels_matrix.shape == (n_samples * n_variants, n_samples * n_variants)

    # Compute cosine similarity matrix, which is just X @ X.T after X is normalized
    # across the feature dimensions (i.e. rows of X)
    embeddings = nn.functional.normalize(embeddings, dim=1)
    sim_matrix = embeddings @ embeddings.T
    # can be commented out after testing
    assert sim_matrix.shape == (n_samples * n_variants, n_samples * n_variants)

    # Discard the main diagonal: exclude self comparison (x_anchor vs. x_anchor)
    n_rows = sim_matrix.shape[0]  # should be batch_size * n_variants
    mask = torch.eye(n_rows, dtype=torch.bool).to(device)
    # labels_matrix and sim_matrix are now both of shape (n_rows, n_rows-1)
    labels_matrix = labels_matrix[~mask].view(n_rows, -1)
    sim_matrix = sim_matrix[~mask].view(n_rows, -1)

    # Select the positive and negative pairs
    # positives: (n_rows, n_variants-1) where n_rows = batch_size * n_variants
    positives = sim_matrix[labels_matrix].view(n_rows, -1)
    # negatives: (n_rows, n_rows-n_variants)
    negatives = sim_matrix[~labels_matrix].view(n_rows, -1)
    # Check shapes (can be commented out after testing)
    assert positives.shape == (n_rows, n_variants - 1)
    assert negatives.shape == (n_rows, n_rows - n_variants)

    # Concatenate positives and negatives to form logits matrix of shape
    # (n_rows, n_rows-1), where logits_matrix[i, j] is the similarity/logit/log-odds
    # of samples i and j being predicted to be from the same frame.
    # The positive pair is always in the left-most column, so the correct label for
    # each row is always 0.
    logits_matrix = torch.cat([positives, negatives], dim=1) / temperature
    labels_index = torch.zeros(n_rows, dtype=torch.long).to(device)

    # Compute cross-entropy loss against the labels
    loss = F.cross_entropy(logits_matrix, labels_index, reduction="mean")
    return loss
