import numpy as np
import torch
import torch.nn as nn
import h5py
import logging
from time import time
from pathlib import Path

from biomechpose.pose_estimation import SimulatedDataSequence
from biomechpose.pose_estimation.contrast_representation import (
    ContrastivePretrainingPipeline,
    ResNetFeatureExtractor,
    ContrastiveProjectionHead,
)
from biomechpose.util import get_hardware_availability


def load_trained_weights(
    feature_extractor: nn.Module,
    projection_head: nn.Module,
    checkpoint_dir: Path,
    epoch: int,
    step: int,
    device: torch.device,
) -> tuple[nn.Module, nn.Module]:
    filename_stem = f"checkpoint_epoch{epoch:03d}_step{step:06d}"
    feature_extractor_path = checkpoint_dir / f"{filename_stem}.feature_extractor.pth"
    projection_head_path = checkpoint_dir / f"{filename_stem}.projection_head.pth"
    if not feature_extractor_path.is_file() or not projection_head_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found in {checkpoint_dir} for epoch {epoch}, step {step}. "
            f"Expected files: {feature_extractor_path} and {projection_head_path}"
        )
    feature_extractor_weights = torch.load(feature_extractor_path, map_location=device)
    projection_head_weights = torch.load(projection_head_path, map_location=device)
    feature_extractor.load_state_dict(feature_extractor_weights)
    projection_head.load_state_dict(projection_head_weights)
    return feature_extractor, projection_head


def predict_for_dataset(
    pipeline: ContrastivePretrainingPipeline,
    dataset: SimulatedDataSequence,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    h_features_all, z_features_all = [], []

    n_batches = 0
    start_time = time()
    for batch in dataset.generate_batches(batch_size):
        # batch: (n_variants * n_frames, n_channels=3, n_rows, n_cols)
        # h_feature, z_features: (n_variants * n_frames, feature_dim)
        h_features, z_features = pipeline.inference(batch)

        # Reshape back to separate variants and frames
        n_samples_this_batch = batch.shape[0] // dataset.n_variants
        h_features = h_features.view(dataset.n_variants, n_samples_this_batch, -1)
        z_features = z_features.view(dataset.n_variants, n_samples_this_batch, -1)

        h_features_all.append(h_features)
        z_features_all.append(z_features)
        n_batches += 1

    walltime = time() - start_time
    logging.info(
        f"Processed dataset {dataset.sim_name} in {n_batches} batches. "
        f"Time taken: {walltime:.2f} seconds."
    )
    return torch.cat(h_features_all, dim=1), torch.cat(z_features_all, dim=1)


def initialize_output_hdf_file(output_path: Path, n_variants: int, n_frames: int):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as f:
        f.create_group("h_features")
        f.create_group("z_features")
        f.attrs["n_variants"] = n_variants
        f.attrs["n_frames"] = n_frames


if __name__ == "__main__":
    ########################### CONFIGURATIONS ############################
    # Sampling configs:
    # Numbers of samples (frames) and variants (synthetic images made by different
    # style transfer models) in each pre-extracted atomic batch
    atomic_batch_nsamples = 32
    atomic_batch_nvariants = 4
    # Batch size for inference. Note that this is equavalent to (n_samples * n_variants)
    # during training because we no longer make the distinction between images coming
    # from different frames/variants during inference. The batch size is also generally
    # bigger because we don't need to save gradients for backpropagation.
    batch_size = 1024

    # Model configs:
    # Directory to data for the training run
    model_data_dir = Path(
        "bulk_data/pose_estimation/contrastive_pretraining/first_run_on_workstation"
    )
    checkpoint_dir = model_data_dir / "checkpoints"
    # Size of projection head
    projection_head_hidden_dim = 512
    projection_head_output_dim = 256

    # Data configs:
    # Image size and number of channels in input images (style transfer outputs)
    image_size = (256, 256)
    n_channels = 3
    # Paths to training and validation data
    synthetic_videos_basedir = Path(
        "bulk_data/style_transfer/production/translated_videos"
    )
    sim_data_basedir = Path("bulk_data/nmf_rendering/")
    style_transfer_models = [  # (model_name, epoch)
        ("ngf16_netGsmallstylegan2_batsize2_lambGAN0.2", 121),
        ("ngf16_netGstylegan2_batsize4_lambGAN0.2", 200),
        ("ngf32_netGstylegan2_batsize2_lambGAN0.5-cont1", 161),
        ("ngf32_netGstylegan2_batsize4_lambGAN0.1", 161),
        ("ngf32_netGstylegan2_batsize4_lambGAN0.5", 141),
        ("ngf32_netGstylegan2_batsize4_lambGAN1.0", 161),
        ("ngf48_netGstylegan2_batsize4_lambGAN0.1", 141),
        ("ngf48_netGstylegan2_batsize2_lambGAN0.1", 141),
    ]

    # Training stage configs:
    checkpoints_to_evaluate = [  # (epoch, step)
        (0, 0),
        (0, 15000),
        (1, 0),
        (1, 15000),
        (2, 0),
        (2, 15000),
        (3, 0),
        (3, 15000),
        (4, 0),
        (4, 15000),
        (5, 0),
    ]

    # Output configs:
    inference_output_dir = model_data_dir / "inference"
    logging_level = logging.INFO
    ######################## END OF CONFIGURATIONS ########################

    # System setup
    get_hardware_availability(check_gpu=True, print_results=True)
    logging.basicConfig(
        level=logging_level, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Initialize datasets
    simulations_to_use = [
        path.parent.parts[-3:]  # get list of (exp_trial, segment, subsegment)
        for path in sim_data_basedir.rglob(
            "BO_Gal4_fly5_trial005/*/*/processed_simulation_data.h5"
        )
    ]
    datasets = []
    for exp_trial, segment, subsegment in sorted(simulations_to_use):
        sim_name = f"{exp_trial}/{segment}/{subsegment}"
        synthetic_video_paths = [
            synthetic_videos_basedir / sim_name / f"translated_{model}_epoch{epoch}.mp4"
            for model, epoch in style_transfer_models
        ]
        dataset = SimulatedDataSequence(synthetic_video_paths, sim_name=sim_name)
        datasets.append(dataset)

    # Initialize models (feature extractor & projection head) and load trained weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor = ResNetFeatureExtractor(pretrained=False).to(device)
    projection_head = ContrastiveProjectionHead(
        input_dim=feature_extractor.output_dim,
        hidden_dim=projection_head_hidden_dim,
        output_dim=projection_head_output_dim,
    ).to(device)

    # Initialize contrastive pretraining pipeline
    # Note: one can also do this with just the raw nn.Module's alone, but the
    # `.inference(batch)` method is just a little more convenient.
    contrastive_pipeline = ContrastivePretrainingPipeline(
        feature_extractor=feature_extractor,
        projection_head=projection_head,
        device=device,
        use_float16=True,
    )

    # Initialize h5 files to save extracted features
    for dataset in datasets:
        output_path = inference_output_dir / dataset.sim_name / "contrastive_latents.h5"
        initialize_output_hdf_file(output_path, dataset.n_variants, dataset.n_frames)

    # Run inference on test data and save outputs
    for i_checkpoint, (epoch, step) in enumerate(checkpoints_to_evaluate):
        logging.info(
            f"Running model from epoch {epoch}, step {step} on {len(datasets)} datasets"
        )

        # Load checkpoint weights
        feature_extractor, projection_head = load_trained_weights(
            feature_extractor,
            projection_head,
            checkpoint_dir,
            epoch=epoch,
            step=step,
            device=device,
        )
        contrastive_pipeline.feature_extractor = feature_extractor
        contrastive_pipeline.projection_head = projection_head

        for i_dataset, dataset in enumerate(datasets):
            # h/z_features: (n_variants, n_frames, feature_dim)
            h_features, z_features = predict_for_dataset(
                contrastive_pipeline, dataset, batch_size=batch_size
            )
            # Save to h5
            output_path = (
                inference_output_dir / dataset.sim_name / "contrastive_latents.h5"
            )
            with h5py.File(output_path, "a") as f:
                f["h_features"].create_dataset(
                    f"epoch{epoch:03d}_step{step:06d}",
                    data=h_features.cpu().numpy().astype(np.float16),
                )
                f["z_features"].create_dataset(
                    f"epoch{epoch:03d}_step{step:06d}",
                    data=z_features.cpu().numpy().astype(np.float16),
                )
