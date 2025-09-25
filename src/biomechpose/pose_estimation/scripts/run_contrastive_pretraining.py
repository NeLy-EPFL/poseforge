import logging
from torch.utils.data import DataLoader
from pathlib import Path

from biomechpose.pose_estimation import AtomicBatchDataset
from biomechpose.pose_estimation.contrast_representation import (
    ContrastivePretrainingPipeline,
    ResNetFeatureExtractor,
    ContrastiveProjectionHead,
)
from biomechpose.util import set_random_seed, get_hardware_availability


if __name__ == "__main__":
    ########################### CONFIGURATIONS ############################
    # Sampling configs:
    # Numbers of samples (frames) and variants (synthetic images made by different
    # style transfer models) in each pre-extracted atomic batch
    atomic_batch_nsamples = 32
    atomic_batch_nvariants = 4
    # Number of different frames to include in each batch. Note that n_variants variants
    # of each frame will be included, so effective batch size = batch_size * n_variants.
    # This must be a multiple of `atomic_epoch_nsamples` in `AtomicBatchDataset`.
    batch_size = 96  # ~8.25GB GPU memory with batch_size=96, n_variants=4
    num_workers = 4

    # Model configs:
    # Whether to use pretrained weights for the feature extractor, or just use the
    # architecture with random initialization
    use_pretrained_backbone = True
    # Size of projection head
    projection_head_hidden_dim = 512
    projection_head_output_dim = 256

    # Data configs:
    # Image size and number of channels in input images (style transfer outputs)
    image_size = (256, 256)
    n_channels = 3
    # Paths to training and validation data
    data_base_dir = Path("bulk_data/pose_estimation/atomic_batches")
    train_data_dirs = [
        data_base_dir / f"BO_Gal4_fly{fly}_trial{trial:03d}"
        for fly in range(1, 5)  # flies 1-4
        for trial in range(1, 6)  # trials 1-5
    ]
    val_data_dirs = [data_base_dir / f"BO_Gal4_fly1_trial001"]

    # Training configs:
    seed = 42  # random seed for reproducibility
    num_epochs = 10  # total number of epochs to train for
    adam_lr = 3e-4  # learning rate for Adam optimizer
    adam_weight_decay = 1e-4  # weight decay for Adam optimizer

    # Output configs:
    output_basedir = Path("bulk_data/pose_estimation/contrastive_pretraining/trial0")
    log_dir = output_basedir / "logs"
    checkpoint_dir = output_basedir / "checkpoints"
    logging_interval = 10  # log training metrics every N steps
    checkpoint_interval = 500  # save model checkpoint every N steps (NOT EPOCHS!)
    validation_interval = 500  # run validation every N steps (NOT EPOCHS!)
    nbatches_per_validation = 300  # number of batches to use for each validation
    logging_level = logging.INFO
    ######################## END OF CONFIGURATIONS ########################

    # System setup
    set_random_seed(seed)
    get_hardware_availability(check_gpu=True, print_results=True)
    logging.basicConfig(
        level=logging_level, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Initialize datasets and dataloaders
    train_ds = AtomicBatchDataset(
        data_dirs=[Path(path) for path in train_data_dirs],
        n_variants=atomic_batch_nvariants,
        image_size=image_size,
        n_channels=n_channels,
    )
    val_ds = AtomicBatchDataset(
        data_dirs=[Path(path) for path in val_data_dirs],
        n_variants=atomic_batch_nvariants,
        image_size=image_size,
        n_channels=n_channels,
    )
    n_atomic_batches_per_batch = batch_size // atomic_batch_nsamples
    assert (
        batch_size % atomic_batch_nsamples == 0
    ), "`batch_size` must be a multiple of `atomic_batch_nsamples`"
    train_loader = DataLoader(
        train_ds,
        batch_size=n_atomic_batches_per_batch,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=n_atomic_batches_per_batch,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Initialize models
    feature_extractor = ResNetFeatureExtractor(pretrained=use_pretrained_backbone)
    logging.info(
        f"Created feature extractor with output dim {feature_extractor.output_dim}"
    )
    projection_head = ContrastiveProjectionHead(
        input_dim=feature_extractor.output_dim,
        hidden_dim=projection_head_hidden_dim,
        output_dim=projection_head_output_dim,
    )
    logging.info("Created projection head")

    # Initialize contrastive learning pipeline
    pipeline = ContrastivePretrainingPipeline(
        feature_extractor=feature_extractor,
        projection_head=projection_head,
        device="cuda",
        use_float16=True,
    )

    # Train models
    pipeline.train(
        training_data_loader=train_loader,
        validation_data_loader=val_loader,
        num_epochs=num_epochs,
        adam_kwargs={"lr": adam_lr, "weight_decay": adam_weight_decay},
        log_dir=log_dir,
        log_interval=logging_interval,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=checkpoint_interval,
        validation_interval=validation_interval,
        nbatches_per_validation=nbatches_per_validation,
    )
