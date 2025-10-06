import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

import biomechpose.pose_estimation.keypoints_3d.config as config
from biomechpose.util import get_hardware_availability
from biomechpose.pose_estimation.data.synthetic import SimulatedDataSequence
from biomechpose.pose_estimation.keypoints_3d import (
    Pose2p5DModel,
    Pose2p5DLoss,
    Pose2p5DPipeline,
)


def _setup_datasets(
    style_transfer_models: list[str],
    simulation_data_basedir: Path,
    synthetic_videos_basedir: Path,
    synthetic_videos_subdirs: list[Path | str] | None,
):
    # Find all simulations to use based on available style transfer videos
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
        sim_data_path = (
            simulation_data_basedir / sim_name / "processed_simulation_data.h5"
        )
        dataset = SimulatedDataSequence(
            synthetic_video_paths, sim_data_path, sim_name=sim_name
        )
        datasets.append(dataset)

    return datasets


def inference_on_dataset(
    dataset: SimulatedDataSequence, pipeline: Pose2p5DPipeline, batch_size: int
):
    all_preds = defaultdict(list)
    all_labels = defaultdict(list)
    for frames, labels in dataset.generate_batches(batch_size=batch_size):
        preds = pipeline.inference(frames)
        for key, value in preds.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            all_preds[key].append(value)
        for key, value in labels.items():
            all_labels[key].append(value.cpu().numpy())

    # Concatenate across batches
    for key, value in all_preds.items():
        if isinstance(value[0], np.ndarray):
            value = np.concatenate(value, axis=0)
            value = value.reshape(
                (dataset.n_variants, dataset.n_frames) + value.shape[1:]
            )
        all_preds[key] = value

    for key in all_labels.keys():
        all_labels[key] = np.concatenate(all_labels[key], axis=0)

    return all_preds, all_labels


def visualize_predictions(
    preds: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    output_dir: Path,
    data_freq: int = 300,
):
    stride_x = preds["heatmap_stride_cols"][0]
    stride_y = preds["heatmap_stride_rows"][0]
    pred_xy_heatmaps = preds["xy_heatmaps"]  # (variants, frames, keypoints, H, W)
    pred_depth_logits = preds["pred_depth"]  # (variants, frames, keypoints, depth_bins)
    pred_xy = preds["pred_xy"]  # (variants, frames, keypoints, 2)
    pred_xy[:, :, :, 0] *= stride_x
    pred_xy[:, :, :, 1] *= stride_y
    pred_depth = preds["pred_depth"]  # (frames, keylabpoints)
    label_xy = labels["keypoint_pos"][:, :, :2]  # (frames, keypoints, 2)
    label_depth = labels["keypoint_pos"][:, :, 2]  # (frames, keypoints)
    n_variants, n_frames, n_keypoints, _, _ = pred_xy_heatmaps.shape

    fig, axes = plt.subplots(
        n_keypoints, 3, figsize=(3 * 1.5, n_keypoints * 2), tight_layout=True
    )
    t_grid = np.arange(n_frames) / data_freq
    for i_keypoint in range(n_keypoints):
        for i_dim, dim in enumerate(["x", "y", "depth"]):
            ax = axes[i_keypoint, i_dim]
            if i_dim < 2:
                # XY
                pred = pred_xy[:, :, i_keypoint, i_dim]
                label = label_xy[:, i_keypoint, i_dim]
            else:
                # Depth
                pred = pred_depth[:, :, i_keypoint]
                label = label_depth[:, i_keypoint]
            for i_variant in range(n_variants):
                ax.plot(t_grid, pred[i_variant, :], linewidth=1)
            ax.plot(t_grid, label, color="black", linewidth=2)
    fig.savefig(output_dir / "xyz_timeseries.png")


def test_keypoints3d_models(
    style_transfer_models: list[str],
    simulation_data_basedir: str,
    synthetic_videos_basedir: str,
    synthetic_videos_subdirs: list[str] | None,
    model_architecture_config_path: str,
    model_checkpoint_path: str,
    loss_config_path: str | None,
    batch_size: int,
    output_basedir: str,
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
    # model.load_weights_from_config(
    #     config.ModelWeightsConfig(model_weights=model_checkpoint_path)
    # )
    model.load_state_dict(
        torch.load(model_checkpoint_path)["model"]
    )  # TODO: remove this hack
    if loss_config_path:
        loss_func = Pose2p5DLoss.create_from_config(loss_config_path)
    else:
        loss_func = None
    pipeline = Pose2p5DPipeline(model, loss_func, device="cuda", use_float16=True)

    # Set up datasets
    datasets = _setup_datasets(
        style_transfer_models,
        Path(simulation_data_basedir),
        Path(synthetic_videos_basedir),
        synthetic_videos_subdirs,
    )
    if len(datasets) == 0:
        raise RuntimeError("No datasets found for testing")
    print(f"Found {len(datasets)} datasets for testing.")

    # Run inference for each dataset
    for dataset in datasets:
        print(f"Running inference on dataset {dataset.sim_name}")
        preds, labels = inference_on_dataset(dataset, pipeline, batch_size)
        preds["pred_depth"] = preds["pred_depth"] - 100  # TODO: Remove this hack
        output_dir = Path(output_basedir) / dataset.sim_name.replace("/", "_")
        output_dir.mkdir(parents=True, exist_ok=True)
        visualize_predictions(preds, labels, output_dir)
        break


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
    synthetic_videos_subdirs = [
        "bulk_data/style_transfer/production/translated_videos/BO_Gal4_fly5_trial005"
    ]
    model_architecture_config_path = "bulk_data/pose_estimation/keypoints3d/trial_20250103a/configs/model_architecture_config.yaml"
    model_checkpoint_path = "bulk_data/pose_estimation/keypoints3d/trial_20250103a/checkpoints/epoch9_step50000.pt"
    loss_config_path = (
        "bulk_data/pose_estimation/keypoints3d/trial_20250103a/configs/loss_config.yaml"
    )
    batch_size = 32
    output_basedir = "bulk_data/pose_estimation/keypoints3d/trial_20250103a/inference/"
    test_keypoints3d_models(
        style_transfer_models=style_transfer_models,
        simulation_data_basedir=simulation_data_basedir,
        synthetic_videos_basedir=synthetic_videos_basedir,
        synthetic_videos_subdirs=synthetic_videos_subdirs,
        model_architecture_config_path=model_architecture_config_path,
        model_checkpoint_path=model_checkpoint_path,
        loss_config_path=loss_config_path,
        batch_size=batch_size,
        output_basedir=output_basedir,
    )
