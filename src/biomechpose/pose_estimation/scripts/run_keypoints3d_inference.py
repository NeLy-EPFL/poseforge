import torch
import numpy as np
import h5py
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from pvio.torch import VideoCollectionDataset, VideoCollectionDataLoader

from biomechpose.pose_estimation.keypoints_3d import Pose2p5DModel, Pose2p5DPipeline
from biomechpose.pose_estimation.keypoints_3d.config import ModelWeightsConfig
from biomechpose.pose_estimation.camera import CameraToWorldMapper
from biomechpose.simulate_nmf.constants import keypoint_segments_canonical
from biomechpose.util import get_hardware_availability


if __name__ == "__main__":
    input_basedir = Path("bulk_data/behavior_images/spotlight_aligned_and_cropped/")
    model = "trial_20251007a"
    model_dir = Path("bulk_data/pose_estimation/keypoints3d") / model
    output_basedir = model_dir / "production"
    architecture_config_path = model_dir / "configs/model_architecture_config.yaml"
    checkpoint_path = model_dir / "checkpoints/epoch13_step9167.model.pth"
    batch_size = 40
    n_workers = 4
    camera_pos = (0.0, 0.0, -100.0)  # mm
    camera_fov_deg = 5.0  # degrees
    camera_rendering_size = (464, 464)  # pixels
    camera_rotation_euler = (0, np.pi, -np.pi / 2)  # radians

    # Find all trials to process
    input_trials = list(input_basedir.glob("*/model_prediction/not_flipped/"))
    print(f"Found {len(list(input_trials))} trials to process")

    # System setup
    hardware_avail = get_hardware_availability(check_gpu=True, print_results=True)
    if len(hardware_avail["gpus"]) == 0:
        raise RuntimeError("No GPU available for testing")
    torch.backends.cudnn.benchmark = True

    # Create dataset and dataloader
    dataset = VideoCollectionDataset(input_trials, as_image_dirs=True)
    dataloader = VideoCollectionDataLoader(
        dataset, batch_size=batch_size, num_workers=n_workers
    )
    print(f"Found {len(dataset)} frames to process")
    print(
        f"Using batch size {dataloader.batch_size} with {dataloader.num_workers} "
        f"workers. This will generate {len(dataloader)} batches."
    )

    # Create model and learning pipeline
    model = Pose2p5DModel.create_architecture_from_config(architecture_config_path)
    model.load_weights_from_config(ModelWeightsConfig(model_weights=checkpoint_path))
    pipeline = Pose2p5DPipeline(model, device="cuda", use_float16=True)

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
            xyz = pred_world_xyz[i, :, :]
            video_path = batch["video_paths"][i]
            frame_idx = batch["frame_indices"][i]
            results[video_path][frame_idx] = xyz
    print("Inference complete")

    # Save results
    for video_path, result_by_frame in results.items():
        trial_name = video_path.parent.parent.name
        output_dir = output_basedir / trial_name
        output_dir.mkdir(parents=True, exist_ok=True)
        results_li = [result_by_frame[i] for i in range(len(result_by_frame))]
        results_stack = np.stack(results_li)
        frame_ids = [
            int(p.stem.split("_")[1]) for p in dataset.frame_sortings[video_path]
        ]
        with h5py.File(output_dir / "keypoints3d.h5", "w") as f:
            f.create_dataset("frame_ids", data=frame_ids, dtype=np.int32)
            ds = f.create_dataset("keypoints_pos", data=results_stack, dtype=np.float16)
            ds.attrs["keypoints"] = keypoint_segments_canonical
            ds.attrs["units"] = "mm"
        print(f"Wrote results to {output_dir / 'keypoints3d.h5'}")
