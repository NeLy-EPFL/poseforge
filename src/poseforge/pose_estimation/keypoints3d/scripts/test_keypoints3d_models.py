import torch
import logging
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Callable
from torchsummary import summary

import poseforge.pose_estimation.keypoints3d.config as config
from poseforge.util import get_hardware_availability
from poseforge.pose_estimation.data.synthetic import SimulatedDataSequence
from poseforge.pose_estimation.keypoints3d import (
    Pose2p5DModel,
    Pose2p5DLoss,
    Pose2p5DPipeline,
)
from poseforge.pose_estimation.camera import CameraToWorldMapper
from poseforge.pose_estimation.keypoints3d.visualizer import (
    SynthDataKeypoints3DVisualizer,
)


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def _setup_datasets(
    style_transfer_models: list[str],
    simulation_data_basedir: Path,
    synthetic_videos_basedir: Path,
    original_image_size: tuple[int, int] | None,
    filter_func: Callable[[str, str, str], bool] = lambda trial, seg, subseg: True,
):
    # Find all simulations to use based on available style transfer videos
    simulations_to_use = []
    for path in synthetic_videos_basedir.rglob(
        f"translated_{style_transfer_models[0]}.mp4"
    ):
        exp_trial, segment, subsegment = path.absolute().parent.parts[-3:]
        if not filter_func(exp_trial, segment, subsegment):
            continue  # skip if not in filter
        simulations_to_use.append((exp_trial, segment, subsegment))

    # Create dataset objects based on identified files
    datasets = []
    for exp_trial, segment, subsegment in sorted(simulations_to_use):
        sim_name = f"{exp_trial}/{segment}/{subsegment}"
        synthetic_video_paths = [
            synthetic_videos_basedir / sim_name / f"translated_{model}.mp4"
            for model in style_transfer_models
        ]
        sim_data_path = (
            simulation_data_basedir / sim_name / "processed_simulation_data.h5"
        )
        dataset = SimulatedDataSequence(
            synthetic_video_paths,
            sim_data_path,
            sim_name=sim_name,
            original_image_size=original_image_size,
        )
        datasets.append(dataset)

    logging.info(f"Found {len(datasets)} datasets for testing.")
    return datasets


def inference_on_dataset(
    dataset: SimulatedDataSequence, pipeline: Pose2p5DPipeline, batch_size: int
):
    all_preds = defaultdict(list)
    all_labels = defaultdict(list)

    for frames, labels in dataset.generate_batches(batch_size=batch_size):
        preds = pipeline.inference(frames)

        # Un-interleave predictions: from (n_variants * n_frames_in_batch, ...)
        # to (n_variants, n_frames_in_batch, ...)
        current_batch_frames = frames.shape[0] // dataset.n_variants
        for key, value in preds.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            if isinstance(value, np.ndarray) and len(value.shape) > 0:
                # Reshape from interleaved format back to (variants, frames, ...)
                value = value.reshape(
                    (dataset.n_variants, current_batch_frames, *value.shape[1:])
                )
            all_preds[key].append(value)

        for key, value in labels.items():
            all_labels[key].append(value.cpu().numpy())

    # Concatenate across batches along the frame dimension
    for key, value in all_preds.items():
        if isinstance(value[0], np.ndarray) and len(value[0].shape) > 0:
            # Concatenate along frame dimension (axis=1)
            value = np.concatenate(value, axis=1)
        all_preds[key] = value

    for key in all_labels.keys():
        all_labels[key] = np.concatenate(all_labels[key], axis=0)

    return all_preds, all_labels


def test_keypoints3d_models(
    style_transfer_models: list[str],
    simulation_data_basedir: str,
    synthetic_videos_basedir: str,
    model_architecture_config_path: str,
    model_checkpoint_path: str,
    loss_config_path: str | None,
    batch_size: int,
    original_image_size: tuple[int, int] | None,
    output_basedir: str,
    camera_distance: float = 100.0,
    camera_rotation_euler: tuple[float, float, float] = (0, np.pi, -np.pi / 2),
    camera_fov_deg: float = 5.0,
    plotting_n_workers: int = -2,
    filter_func: Callable[[str, str, str], bool] = lambda trial, seg, subseg: True,
):
    # System setup
    hardware_avail = get_hardware_availability(check_gpu=True, print_results=True)
    if len(hardware_avail["gpus"]) == 0:
        raise RuntimeError("No GPU available for testing")
    torch.backends.cudnn.benchmark = True

    # Load model, loss function, and learning pipeline
    model = Pose2p5DModel.create_architecture_from_config(
        model_architecture_config_path
    )
    model.load_weights_from_config(
        config.ModelWeightsConfig(model_weights=model_checkpoint_path)
    )
    print("========== Model Summary ==========")
    summary(model, input_size=(3, *model.feature_extractor.input_size), device="cpu")

    # Set up loss function if provided
    if loss_config_path:
        loss_func = Pose2p5DLoss.create_from_config(loss_config_path)
    else:
        loss_func = None
    pipeline = Pose2p5DPipeline(model, loss_func, device="cuda", use_float16=True)
    assert next(model.parameters()).is_cuda, "Model is not on GPU"

    # Set up datasets
    datasets = _setup_datasets(
        style_transfer_models,
        Path(simulation_data_basedir),
        Path(synthetic_videos_basedir),
        original_image_size=original_image_size,
        filter_func=filter_func,
    )
    if len(datasets) == 0:
        raise RuntimeError("No datasets found for testing")
    print(f"Found {len(datasets)} datasets for testing.")

    # Set up camera mapper
    cam_mapper = CameraToWorldMapper(
        camera_pos=(0.0, 0.0, -camera_distance),
        camera_fov_deg=camera_fov_deg,
        rendering_size=(original_image_size[0], original_image_size[1]),
        rotation_euler=camera_rotation_euler,
    )

    # Run inference for each dataset
    for dataset in datasets:
        output_dir = Path(output_basedir) / dataset.sim_name.replace("/", "_")
        output_dir.mkdir(parents=True, exist_ok=True)
        pred_path = output_dir / "predictions.npz"
        label_path = output_dir / "labels.npz"
        if pred_path.is_file() and label_path.is_file():
            print(
                f"Found existing predictions and labels "
                f"for dataset {dataset.sim_name}; loading from disk."
            )
            preds = np.load(pred_path)
            labels = np.load(label_path)
        else:
            print(f"Running inference on dataset {dataset.sim_name}")
            preds, labels = inference_on_dataset(dataset, pipeline, batch_size)
            preds["pred_world_xyz"] = cam_mapper(preds["pred_xy"], preds["pred_depth"])
            labels["keypoint_pos_world_xyz"] = cam_mapper(
                labels["keypoint_pos"][:, :, :2], labels["keypoint_pos"][:, :, 2]
            )
            np.savez_compressed(pred_path, **preds)
            np.savez_compressed(label_path, **labels)

        labels_metadata = dataset.get_sim_data_metadata()
        keypoints_order = labels_metadata["keypoint_pos"]["keys"]
        exp_trial, segment, subsegment = dataset.sim_name.split("/")
        visualizer = SynthDataKeypoints3DVisualizer(
            preds,
            labels,
            keypoints_order,
            output_dir,
            nmf_rendering_basedir=simulation_data_basedir,
            synthetic_videos_basedir=synthetic_videos_basedir,
            style_transfer_models=style_transfer_models,
            exp_trial=exp_trial,
            segment=segment,
            subsegment=subsegment,
            n_workers=plotting_n_workers,
        )
        visualizer.plot_keypoints_over_time(output_dir / "keypoint_pos_timeseries.png")
        visualizer.make_summary_video(output_dir / "keypoint_3d_visualization.mp4")


if __name__ == "__main__":
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
    simulation_data_basedir = "bulk_data/nmf_rendering"
    synthetic_videos_basedir = "bulk_data/style_transfer/production/translated_videos/"
    model_dir = Path("bulk_data/pose_estimation/keypoints3d/trial_20251013b/")
    model_architecture_config_path = str(
        model_dir / "configs/model_architecture_config.yaml"
    )
    loss_config_path = str(model_dir / "configs/loss_config.yaml")
    model_checkpoint_path = str(model_dir / "checkpoints/epoch8_step5000.model.pth")
    output_basedir = model_dir / "inference"
    batch_size = 32
    original_image_size = (464, 464)
    simulations = {
        ("BO_Gal4_fly5_trial005", "segment_003", "subsegment_002"),
    }
    test_keypoints3d_models(
        style_transfer_models=style_transfer_models,
        simulation_data_basedir=simulation_data_basedir,
        synthetic_videos_basedir=synthetic_videos_basedir,
        model_architecture_config_path=model_architecture_config_path,
        model_checkpoint_path=model_checkpoint_path,
        loss_config_path=loss_config_path,
        batch_size=batch_size,
        original_image_size=original_image_size,
        output_basedir=output_basedir,
        plotting_n_workers=-2,  # Use all cores except 1 for parallel processing
        filter_func=lambda trial, seg, subseg: (trial, seg, subseg) in simulations,
    )
