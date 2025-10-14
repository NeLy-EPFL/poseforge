import logging

logging_level = logging.INFO
logging.basicConfig(
    level=logging_level, format="%(asctime)s - %(levelname)s - %(message)s"
)

import numpy as np
import torch
import h5py
from torchsummary import summary
from collections import defaultdict
from time import time
from pathlib import Path

from poseforge.pose_estimation.contrast import ContrastivePretrainingModel
from poseforge.pose_estimation.data.synthetic import SimulatedDataSequence
from poseforge.pose_estimation.contrast import ContrastivePretrainingPipeline
from poseforge.util import get_hardware_availability


def load_model(
    model_dir: Path,
    training_stage: str,
    print_summary: bool = False,
    input_size: tuple[int, int, int] = (3, 256, 256),
) -> ContrastivePretrainingModel:
    """Initialize models (feature extractor & projection head) and load
    trained weights.

    Args:
        model_dir (Path): Directory containing model checkpoints and config
            files.
        training_stage (Path): Training stage identifier (can be
            e.g., "epoch000_step015000" or "untrained").
        print_summary (bool): Whether to print model summaries.
        input_size (tuple[int, int, int]): Input size for model summary.

    Returns:
        Initialized ContrastivePretrainingModel with loaded weights.
    """
    checkpoint_dir = Path(model_dir) / "checkpoints"
    configs_dir = Path(model_dir) / "configs"
    contrastive_learning_model = (
        ContrastivePretrainingModel.create_architecture_from_config(
            configs_dir / "model_architecture_config.yaml"
        )
    )

    # Load weights
    if training_stage != "untrained":
        feature_extractor_checkpoint_path = (
            checkpoint_dir / f"checkpoint_{training_stage}.feature_extractor.pth"
        )
        projection_head_checkpoint_path = (
            checkpoint_dir / f"checkpoint_{training_stage}.projection_head.pth"
        )
        contrastive_learning_model.feature_extractor.load_state_dict(
            torch.load(feature_extractor_checkpoint_path, map_location="cpu")
        )
        contrastive_learning_model.projection_head.load_state_dict(
            torch.load(projection_head_checkpoint_path, map_location="cpu")
        )

    if print_summary:
        print("========== Feature Extractor Summary ==========")
        summary(
            contrastive_learning_model.feature_extractor,
            input_size=input_size,
            device="cpu",
        )
        print("=========== Projection Head Summary ===========")
        summary(
            contrastive_learning_model.projection_head,
            input_size=(contrastive_learning_model.feature_extractor.output_channels,),
            device="cpu",
        )

    return contrastive_learning_model


def predict_for_dataset(
    pipeline: ContrastivePretrainingPipeline,
    dataset: SimulatedDataSequence,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    pred_dict_all = defaultdict(list)
    n_batches = 0
    start_time = time()
    for frames, _ in dataset.generate_batches(batch_size):
        # batch: (n_variants * n_frames, n_channels=3, n_rows, n_cols)
        # h_features: (n_variants * n_frames, n_channels=512, *output_feature_map_size)
        # h_features_pooled: (n_variants * n_frames, feature_dim)
        # z_features: (n_variants * n_frames, feature_dim)
        pred_dict = pipeline.inference(frames)
        for key, tensor in pred_dict.items():
            # Reshape back to separate variants and frames
            shape = tensor.shape
            total_samples_x_variants = shape[0]
            n_samples_this_batch = total_samples_x_variants // dataset.n_variants
            assert total_samples_x_variants % dataset.n_variants == 0
            tensor = tensor.view(dataset.n_variants, n_samples_this_batch, *shape[1:])
            pred_dict_all[key].append(tensor)
        n_batches += 1

    walltime = time() - start_time
    logging.info(
        f"Processed dataset {dataset.sim_name} in {n_batches} batches. "
        f"Time taken: {walltime:.2f} seconds."
    )
    return {key: torch.cat(tensors, dim=1) for key, tensors in pred_dict_all.items()}


def initialize_output_hdf_file(output_path: Path, n_variants: int, n_frames: int):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as f:
        f.create_group("h_features_pooled")
        f.create_group("z_features")
        f.attrs["n_variants"] = n_variants
        f.attrs["n_frames"] = n_frames


def run_feature_extractor_inference(
    synthetic_videos_basedir: str,
    pretraining_model_dir: str,
    training_stages: list[str],
    style_transfer_models: list[str],
    batch_size: int = 1024,
    synthetic_videos_subdirs: list[str] | None = None,
):
    """Run inference using a trained contrastive pretraining model to
    extract features from synthetic videos."""
    # System setup
    avail = get_hardware_availability(check_gpu=True, print_results=False)
    if len(avail["gpus"]) == 0:
        raise RuntimeError("No GPU available. Cannot run inference.")
    device = torch.device("cuda")

    # Convert paths to Path objects
    synthetic_videos_basedir = Path(synthetic_videos_basedir)
    pretraining_model_dir = Path(pretraining_model_dir)
    inference_output_dir = pretraining_model_dir / "inference"

    # Find all simulations to process
    if synthetic_videos_subdirs is None:
        synthetic_videos_subdirs = list(synthetic_videos_basedir.glob("*"))
    else:
        synthetic_videos_subdirs = [Path(subdir) for subdir in synthetic_videos_subdirs]
    simulations_to_use = []
    for subdir in synthetic_videos_subdirs:
        for path in subdir.rglob(f"translated_{style_transfer_models[0]}.mp4"):
            # sim_info: (exp_trial, segment, subsegment)
            sim_info = path.absolute().parent.parts[-3:]
            simulations_to_use.append(sim_info)
    # Create dataset objects based on identified files
    datasets = []
    for exp_trial, segment, subsegment in sorted(simulations_to_use):
        sim_name = f"{exp_trial}/{segment}/{subsegment}"
        synthetic_video_paths = [
            synthetic_videos_basedir / sim_name / f"translated_{model}.mp4"
            for model in style_transfer_models
        ]
        dataset = SimulatedDataSequence(synthetic_video_paths, sim_name=sim_name)
        datasets.append(dataset)

    # Initialize h5 files to save extracted features
    for dataset in datasets:
        output_path = inference_output_dir / dataset.sim_name / "contrastive_latents.h5"
        initialize_output_hdf_file(output_path, dataset.n_variants, dataset.n_frames)

    # Run inference on test data and save outputs
    for stage_idx, training_stage in enumerate(training_stages):
        logging.info(
            f"Running model from epoch {training_stage} on {len(datasets)} datasets"
        )

        # Load model and initialize pipeline
        contrastive_learning_model = load_model(
            pretraining_model_dir, training_stage, print_summary=(stage_idx == 0)
        ).to(device)
        pipeline = ContrastivePretrainingPipeline(
            contrastive_learning_model, device=device, use_float16=True
        )

        for dataset in datasets:
            pred_dict = predict_for_dataset(pipeline, dataset, batch_size=batch_size)
            h_features_pooled = pred_dict["h_features_pooled"]
            z_features = pred_dict["z_features"]

            # Save to h5
            output_path = (
                inference_output_dir / dataset.sim_name / "contrastive_latents.h5"
            )
            with h5py.File(output_path, "a") as f:
                f["h_features_pooled"].create_dataset(
                    training_stage,
                    data=h_features_pooled.cpu().numpy().astype(np.float16),
                )
                f["z_features"].create_dataset(
                    training_stage, data=z_features.cpu().numpy().astype(np.float16)
                )


if __name__ == "__main__":
    import tyro

    # tyro.cli(
    #     run_feature_extractor_inference,
    #     prog=f"python {Path(__file__).name}",
    #     description="Run inference using a contrastively pretrained feature extractor.",
    # )

    # Example call using function directly (no CLI)
    training_stages = ["untrained", "epoch000_step002000"]
    style_transfer_models = [
        "ngf16_netGsmallstylegan2_batsize2_lambGAN0.2_epoch121",
        "ngf16_netGstylegan2_batsize4_lambGAN0.2_epoch200",
        "ngf32_netGstylegan2_batsize2_lambGAN0.5-cont1_epoch161",
        "ngf32_netGstylegan2_batsize4_lambGAN0.1_epoch161",
        "ngf32_netGstylegan2_batsize4_lambGAN0.5_epoch141",
        "ngf32_netGstylegan2_batsize4_lambGAN1.0_epoch161",
        "ngf48_netGstylegan2_batsize2_lambGAN0.1_epoch141",
        "ngf48_netGstylegan2_batsize4_lambGAN0.1_epoch141",
    ]
    synthetic_videos_basedir = "bulk_data/style_transfer/production/translated_videos/"
    synthetic_videos_subdirs = [
        "bulk_data/style_transfer/production/translated_videos/BO_Gal4_fly5_trial005/segment_003"
    ]
    pretraining_model_dir = (
        "bulk_data/pose_estimation/contrastive_pretraining/trial_20251011a/"
    )
    batch_size = 1024
    run_feature_extractor_inference(
        synthetic_videos_basedir=synthetic_videos_basedir,
        pretraining_model_dir=pretraining_model_dir,
        training_stages=training_stages,
        style_transfer_models=style_transfer_models,
        batch_size=batch_size,
        synthetic_videos_subdirs=synthetic_videos_subdirs,
    )
