import torch
import numpy as np
import h5py
from torchvision.transforms import Resize
from torchsummary import summary
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from pvio.torch_tools import SimpleVideoCollectionLoader, ImageDirVideo
import argparse
from importlib.resources import files
import yaml

from poseforge.pose.keypoints3d import Pose2p5DModel, Pose2p5DPipeline
from poseforge.pose.keypoints3d.config import ModelWeightsConfig
from poseforge.pose.camera import CameraToWorldMapper
from poseforge.neuromechfly.constants import keypoint_segments_canonical
from poseforge.util import get_hardware_availability


def run_keypoints3d_inference(
    input_basedir: Path,
    model_dir: Path,
    model_checkpoint_path: Path,
    output_basedir: Path | None = None,
    batch_size: int = 512,
    n_workers: int = 16,
    inference_image_size: tuple[int, int] = (256, 256),
    camera_pos: tuple[float, float, float] = (0.0, 0.0, -100.0),
    camera_fov_deg: float = 5.0,
    camera_rendering_size: tuple[int, int] = (464, 464),
    camera_rotation_euler: tuple[float, float, float] = (0, np.pi, -np.pi / 2),
    glob_pattern: str = "fly*",
    ):
    """Run 3D keypoint detection model on all real Spotlight videos and
    save results.

    Args:
        input_basedir: Directory containing subdirectories for each trial,
            each of which should contain a "model_prediction/not_flipped"
            subdirectory with the aligned and cropped images.
        model_dir: Directory containing model architecture and weights
            configuration files.
        model_checkpoint_path: Path to model checkpoint file.
        output_basedir: Directory to save results. If None, will save to
            "production" subdirectory of model_dir.
        batch_size: Batch size to use for inference.
        n_workers: Number of workers to use for data loading.
        inference_image_size: (width, height) to resize input images to
            for inference.
        camera_pos: (x, y, z) position of the camera in world coordinates
            (in mm).
        camera_fov_deg: Camera field of view in degrees (assumed to be the
            same horizontally and vertically).
        camera_rendering_size: (width, height) of the rendered image in
            pixels. Must be square (width == height).
        camera_rotation_euler: (roll, pitch, yaw) rotation of the camera
            in radians. These should be the exact same values that were
            used to create the camera during simulation.
    """
    # Find all trials to process
    input_trials = list(input_basedir.glob(f"{glob_pattern}/model_prediction/not_flipped/"))
    input_trials = [trial for trial in input_trials if len(list(trial.iterdir())) > 0]
    print(f"Found {len(list(input_trials))} trials to process")

    # System setup
    hardware_avail = get_hardware_availability(check_gpu=True, print_results=True)
    if len(hardware_avail["gpus"]) == 0:
        raise RuntimeError("No GPU available for testing")
    torch.backends.cudnn.benchmark = True

    # Create dataset and dataloader
    transform = Resize(inference_image_size)
    dataloader = SimpleVideoCollectionLoader(
        input_trials, transform=transform, batch_size=batch_size, num_workers=n_workers
    )

    print(f"Found {len(dataloader.dataset)} frames to process")
    print(
        f"Using batch size {dataloader.batch_size} with {dataloader.num_workers} "
        f"workers. This will generate ~{len(dataloader)} batches."
    )

    # Create model and learning pipeline
    architecture_config_path = model_dir / "configs/model_architecture_config.yaml"
    model_weights = ModelWeightsConfig(model_weights=model_checkpoint_path)
    model = Pose2p5DModel.create_architecture_from_config(architecture_config_path)
    model.load_weights_from_config(model_weights)
    pipeline = Pose2p5DPipeline(model, device="cuda", use_float16=True)
    print("========== Model Summary ==========")
    summary(pipeline.model, input_size=(3, *inference_image_size), device="cuda")

    # Set up camera mapper
    cam_mapper = CameraToWorldMapper(
        camera_pos, camera_fov_deg, camera_rendering_size, camera_rotation_euler
    )

    # Run inference
    results = defaultdict(lambda: defaultdict(dict))  # results[video_path][frame_idx]
    for batch in tqdm(dataloader):
        # No need to move data to and from the GPU, pipeline will do that
        pred_dict = pipeline.inference(batch["frames"])
        pred_world_xyz = cam_mapper(pred_dict["pred_xy"], pred_dict["pred_depth"])
        for i in range(pred_world_xyz.shape[0]):
            world_xyz = pred_world_xyz[i, :, :]
            camera_xy = pred_dict["pred_xy"][i, :, :]
            camera_depth = pred_dict["pred_depth"][i, :]
            camera_xy_conf = pred_dict["conf_xy"][i, :]
            camera_depth_conf = pred_dict["conf_depth"][i, :]
            video_id = batch["video_indices"][i]
            frame_id = batch["frame_indices"][i]
            results[video_id][frame_id] = (
                world_xyz,
                camera_xy,
                camera_depth,
                camera_xy_conf,
                camera_depth_conf,
            )
    print("Inference complete")

    # Save results
    if output_basedir is None:
        output_basedir = model_dir / "production"
    for video_id, result_by_vir_frame_id in results.items():
        video_obj = dataloader.dataset.videos[video_id]
        assert isinstance(video_obj, ImageDirVideo)
        trial_name = video_obj.path.parent.parent.name
        output_dir = output_basedir / trial_name
        output_dir.mkdir(parents=True, exist_ok=True)

        phy_frame_ids = [
            video_obj.frame_id_vir2phy[vir_frame_id]
            for vir_frame_id in result_by_vir_frame_id.keys()
        ]
        phy_frame_ids.sort()
        sorted_results = []
        for phy_frame_id in phy_frame_ids:
            vir_frame_id = video_obj.frame_id_phy2vir[phy_frame_id]
            res = result_by_vir_frame_id[vir_frame_id]
            sorted_results.append(res)

        with h5py.File(output_dir / "keypoints3d.h5", "w") as f:
            f.create_dataset(
                "frame_ids",
                data=np.array(phy_frame_ids, dtype=np.int32),
                dtype=np.int32,
                compression="gzip",
                shuffle=True,
            )

            world_xyz_stack = np.stack([res[0] for res in sorted_results])
            world_xyz_ds = f.create_dataset(
                "keypoints_world_xyz",
                data=world_xyz_stack,
                dtype=np.float16,
                compression="gzip",
            )
            world_xyz_ds.attrs["keypoints"] = keypoint_segments_canonical
            world_xyz_ds.attrs["units"] = "mm"

            camera_xy_stack = np.stack([res[1] for res in sorted_results])
            camera_xy_ds = f.create_dataset(
                "keypoints_camera_xy",
                data=camera_xy_stack,
                dtype=np.float16,
                compression="gzip",
            )
            camera_xy_ds.attrs["keypoints"] = keypoint_segments_canonical
            camera_xy_ds.attrs["units"] = "pixels"
            camera_xy_ds.attrs["image_size"] = list(inference_image_size)

            camera_depth_stack = np.stack([res[2] for res in sorted_results])
            camera_depth_ds = f.create_dataset(
                "keypoints_camera_depth",
                data=camera_depth_stack,
                dtype=np.float16,
                compression="gzip",
            )
            camera_depth_ds.attrs["keypoints"] = keypoint_segments_canonical
            camera_depth_ds.attrs["units"] = "mm"

            camera_xy_conf_stack = np.stack([res[3] for res in sorted_results])
            camera_xy_conf_ds = f.create_dataset(
                "keypoints_camera_xy_conf",
                data=camera_xy_conf_stack,
                dtype=np.float16,
                compression="gzip",
            )
            camera_xy_conf_ds.attrs["keypoints"] = keypoint_segments_canonical
            camera_xy_conf_ds.attrs["method"] = model.confidence_method

            camera_depth_conf_stack = np.stack([res[4] for res in sorted_results])
            camera_depth_conf_ds = f.create_dataset(
                "keypoints_camera_depth_conf",
                data=camera_depth_conf_stack,
                dtype=np.float16,
                compression="gzip",
            )
            camera_depth_conf_ds.attrs["keypoints"] = keypoint_segments_canonical
            camera_depth_conf_ds.attrs["method"] = model.confidence_method

        print(f"Wrote results to {output_dir / 'keypoints3d.h5'}")

def start():
    parser = argparse.ArgumentParser(
        description="Detect flipped flies in spotlight recordings."
    )
    parser.add_argument(
        "aligned_data_dir",
        type=Path,
        default=Path("bulk_data/spotlight_aligned_and_cropped"),
        help="Base directory containing aligned and cropped spotlight recording trials.",
    )
    parser.add_argument(
        "glob_pattern",
        type=str,
        default="fly*",
        help="Glob pattern to match spotlight trial directories.",
    )
    # get package root path for default config path
    parser.add_argument(
        "--config_path",
        type=Path,
        # path relative to poseforge package root
        default=files("poseforge").joinpath(
            "production/spotlight/config.yaml"
        ),
    )
    # make optional
    parser.add_argument(
        "--segment_model_dir",
        type=Path,
        help="Path to keypoint prediction model directory. If not provided, will be loaded from config file.",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--epoch",
        type=int,
        help="Epoch number of the model checkpoint to use for inference.",
        default=19,
    )
    parser.add_argument(
        "--step",
        type=int,
        help="Step number of the model checkpoint to use for inference.",
        default=9167,
    )
    parser.add_argument(
        "--output_basedir",
        type=Path,
        help="Base directory to save inference results. If not provided, will save to 'production' subdirectory of model directory.",
        default=None,
    )

    args = parser.parse_args()
    
    return args.aligned_data_dir, args.glob_pattern, args.config_path, args.segment_model_dir, args.epoch, args.step, args.output_basedir

if __name__ == "__main__":
    #input_basedir = Path("bulk_data/behavior_images/spotlight_aligned_and_cropped/")
    input_basedir, glob_pattern, config_path, segment_model_dir, epoch, step, output_basedir = start()
    if not segment_model_dir:
        # load from config file
        with open(config_path, "r") as f:
            prod_config = yaml.safe_load(f)
        model_dir = Path(prod_config["keypoints3d"]["checkpoint"]).parent.parent
    else:
        model_dir = segment_model_dir

    print(f"Running inference for epoch {epoch}")
    checkpoint_path = model_dir / f"checkpoints/epoch{epoch}_step{step}.model.pth"
    if output_basedir is  None:
        output_basedir = model_dir / f"production/epoch{epoch}_step{step}/"
    else:
        output_basedir = output_basedir / f"keypoints3d/epoch{epoch}_step{step}/"
    output_basedir.mkdir(parents=True, exist_ok=True)

    run_keypoints3d_inference(
        input_basedir, model_dir, checkpoint_path, output_basedir=output_basedir, glob_pattern=glob_pattern
    )
