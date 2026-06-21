import torch
import h5py
from pathlib import Path
from torchvision.transforms import Resize
from torchsummary import summary
from tqdm import tqdm
from pvio.torch_tools import SimpleVideoCollectionLoader
import argparse
from importlib.resources import files
import yaml
import re
from typing import Callable, Any

import poseforge.pose.bodyseg.config as config
from poseforge.pose.bodyseg import BodySegmentationModel, BodySegmentationPipeline
from poseforge.util.sys import get_hardware_availability
from poseforge.util.data import OutputBuffer



def compute_confidence_from_probs(probs: torch.Tensor, method: str = "entropy") -> torch.Tensor:
    # probs shape: [..., n_classes, H, W]
    n_classes = probs.shape[-3]
    if method == "entropy":
        entropy = -(probs * torch.log(probs.clamp_min(1e-6))).sum(dim=-3)
        n_classes_tensor = torch.tensor(n_classes, dtype=entropy.dtype, device=entropy.device)
        normalized_entropy = entropy / torch.log(n_classes_tensor)
        confidence = 1.0 - normalized_entropy
    elif method == "peak":
        confidence, _ = torch.max(probs, dim=-3)
    else:
        raise ValueError(f"Unknown confidence method: {method}")
    return confidence


def run_bodyseg_inference_generic(
    input_basedir: Path,
    model_dir: Path,
    model_checkpoint_path: Path,
    output_basedir: Path | None = None,
    batch_size: int = 512,
    n_workers: int = 16,
    inference_image_size: tuple[int, int] = (256, 256),
    class_labels: list[str] | None = None,
    output_buffer_log_interval: int = 10,
    glob_pattern: str = "fly*",
    output_filename: str = "bodyseg_pred.h5",
    process_batch_func: Callable[[BodySegmentationPipeline, dict], list[Any]] | None = None,
    save_predictions_func: Callable[[h5py.File, BodySegmentationPipeline, list[Any], Any], None] | None = None,
):
    # System setup
    hardware_avail = get_hardware_availability(check_gpu=True, print_results=True)
    if len(hardware_avail["gpus"]) == 0:
        raise RuntimeError("No GPU available for testing")
    torch.backends.cudnn.benchmark = True

    # Find all trials to process
    input_trials = list(input_basedir.glob(f"{glob_pattern}/model_prediction/not_flipped/"))
    print(f"Found {len(list(input_trials))} trials to process")
    input_trials = [trial for trial in input_trials if len(list(trial.iterdir())) > 0]
    print(f"{len(input_trials)} trials have images to process")

    # Create dataset and dataloader
    transform = Resize(inference_image_size)
    dataloader = SimpleVideoCollectionLoader(
        input_trials, transform=transform, batch_size=batch_size, num_workers=n_workers
    )
    print(f"Found {len(dataloader.dataset)} frames to process")
    print(
        f"Using batch size {dataloader.batch_size} with {dataloader.num_workers} "
        f"workers. This will generate {len(dataloader)} batches."
    )

    # Create model and learning pipeline
    architecture_config_path = model_dir / "configs/model_architecture_config.yaml"
    print(f"Loading model architecture from {architecture_config_path}")
    model = BodySegmentationModel.create_architecture_from_config(
        architecture_config_path
    ).cuda()
    model_weights = config.ModelWeightsConfig(model_weights=model_checkpoint_path)
    print(f"Loading model weights from {model_weights}")
    model.load_weights_from_config(model_weights)
    summary(model, (3, *inference_image_size))
    pipeline = BodySegmentationPipeline(model, device="cuda", use_float16=True)

    # Make an output buffer - output data for multiple videos will arrive out of sync
    def save_predictions(input_video_idx, data_items):
        video_obj = dataloader.dataset.videos[input_video_idx]
        input_video_path = video_obj.path
        exp_trial_name = "_".join(input_video_path.parts[-3:])
        out_dir = output_basedir / exp_trial_name
        out_dir.mkdir(parents=True, exist_ok=True)
        with h5py.File(out_dir / f"bodyseg_pred.h5", "w") as f:
            pred_segmaps = torch.stack([x[0] for x in data_items], dim=0).cpu().numpy()
            ds = f.create_dataset(
                "pred_segmap",
                data=pred_segmaps,
                dtype="uint8",
                compression="gzip",
                shuffle=True,
            )
            if class_labels is not None:
                ds.attrs["class_labels"] = class_labels
            else:
                ds.attrs["class_labels"] = pipeline.class_labels
            confs = torch.stack([x[1] for x in data_items], dim=0).cpu().numpy()
            ds = f.create_dataset(
                "pred_confidence",
                data=confs,
                dtype="uint8",
                compression="gzip",
                shuffle=True,
            )
            # Confidence is predicted in 0-1, but we store it in 0-100 as uint8
            ds.attrs["scale"] = 100
            ds.attrs["method"] = model.confidence_method
            frame_ids = [
                int(p.stem.split("_")[1])
                for p in video_obj.phy_frame_id_to_path.values()
            ]
            # These are the actual, raw frame IDs from the original video assigned by
            # the Spotlight recording software. They may not be contiguous because
            # frames where the fly is upside down or too close to the edge, etc. are
            # already removed.
            f.create_dataset(
                "frame_ids",
                data=frame_ids,
                dtype="int",
                compression="gzip",
                shuffle=True,
            )

    buckets_and_sizes = {
        i: n_frames for i, n_frames in enumerate(dataloader.dataset.n_frames_by_video)
    }
    output_buffer = OutputBuffer(
        buckets_and_expected_sizes=buckets_and_sizes,
        closing_func=save_predictions,
    )

    # Run inference
    for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        # No need to move data to and from the GPU, pipeline will do that
        if process_batch_func is not None:
            data_items = process_batch_func(pipeline, batch)
        else:
            pred_dict = pipeline.inference(batch["frames"])
            logits = pred_dict["logits"]
            pred_seg = torch.argmax(logits, dim=1).to(torch.uint8).detach().cpu()
            confidence = (pred_dict["confidence"] * 100).to(torch.uint8).detach().cpu()
            data_items = [(pred_seg[i, :, :], confidence[i, :, :]) for i in range(logits.shape[0])]

        for i in range(len(data_items)):
            output_buffer.add_data(
                bucket=batch["video_indices"][i],
                index=batch["frame_indices"][i],
                data=data_items[i],
            )
        if (batch_idx + 1) % output_buffer_log_interval == 0:
            print(
                f"{batch_idx + 1}/{len(dataloader)} batches - "
                f"{output_buffer.n_open_buckets} partially processed videos, "
                f"{output_buffer.n_data_total} total frames in buffer"
            )

    assert output_buffer.n_data_total == 0
    assert output_buffer.n_open_buckets == 0
    print("Inference complete")


def test_bodyseg_model(
    input_basedir: Path,
    model_dir: Path,
    model_checkpoint_path: Path,
    contrastive_checkpoint_path: Path | None = None,
    output_basedir: Path | None = None,
    batch_size: int = 512,
    n_workers: int = 16,
    inference_image_size: tuple[int, int] = (256, 256),
    output_buffer_log_interval: int = 10,
    glob_pattern: str = "fly*",
):
    run_bodyseg_inference_generic(
        input_basedir=input_basedir,
        model_dir=model_dir,
        model_checkpoint_path=model_checkpoint_path,
        contrastive_checkpoint_path=contrastive_checkpoint_path,
        output_basedir=output_basedir,
        batch_size=batch_size,
        n_workers=n_workers,
        inference_image_size=inference_image_size,
        output_buffer_log_interval=output_buffer_log_interval,
        glob_pattern=glob_pattern,
        output_filename="bodyseg_pred.h5",
    )


def start():
    parser = argparse.ArgumentParser(
        description="Run body segmentation inference on spotlight recordings."
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
    parser.add_argument(
        "--config_path",
        type=Path,
        default=files("poseforge").joinpath(
            "production/spotlight/config.yaml"
        ),
        help="Path to config file containing model paths and parameters.",
    )
    parser.add_argument(
        "--output_basedir",
        type=Path,
        help="Base directory to save output predictions. If not provided, will be saved in the model directory under production/epoch{epoch}_step{step}/",
        required=False,
        default=None,
    )

    args = parser.parse_args()
    
    return args.aligned_data_dir, args.glob_pattern, args.config_path, args.output_basedir

if __name__ == "__main__":
    # parse paths
    input_basedir, glob_pattern, config_path, output_basedir = start()
    
    # load from config file
    with open(config_path, "r") as f:
        prod_config = yaml.safe_load(f)
    
    # Extract epoch and step from checkpoint filename
    checkpoint_path = Path(prod_config["bodyseg"]["checkpoint"])
    match = re.search(r"epoch(\d+)_step(\d+)", checkpoint_path.stem)
    if not match:
        raise ValueError(
            f"Could not extract epoch and step from checkpoint path: {checkpoint_path}"
        )
    epoch, step = int(match.group(1)), int(match.group(2))

    model_dir = checkpoint_path.parent.parent
    batch_size = prod_config["bodyseg"]["batch_size"]
    n_workers = prod_config.get("common", {}).get("n_workers", prod_config["bodyseg"].get("n_workers", 16))
    inference_image_size = tuple(prod_config.get("common", {}).get("inference_image_size") or \
                                 prod_config["bodyseg"]["inference_image_size"])
    class_labels = prod_config.get("common", {}).get("class_labels")
    output_buffer_log_interval = prod_config["bodyseg"]["output_buffer_log_interval"]

    if output_basedir is None:
        output_basedir = model_dir / f"production/epoch{epoch}_step{step}/"
    else: 
        output_basedir = output_basedir / f"bodyseg/epoch{epoch}_step{step}/"
    output_basedir.mkdir(parents=True, exist_ok=True)

    test_bodyseg_model(
        input_basedir=input_basedir,
        model_dir=model_dir,
        model_checkpoint_path=checkpoint_path,
        output_basedir=output_basedir,
        batch_size=batch_size,
        n_workers=n_workers,
        inference_image_size=inference_image_size,
        class_labels=class_labels,
        output_buffer_log_interval=output_buffer_log_interval,
        glob_pattern=glob_pattern,
    )
