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

import poseforge.pose.bodyseg.config as config
from poseforge.pose.bodyseg import BodySegmentationModel, BodySegmentationPipeline
from poseforge.util.sys import get_hardware_availability
from poseforge.util.data import OutputBuffer


def test_bodyseg_model(
    input_basedir: Path,
    model_dir: Path,
    model_checkpoint_path: Path,
    output_basedir: Path | None = None,
    batch_size: int = 512,
    n_workers: int = 16,
    inference_image_size: tuple[int, int] = (256, 256),
    output_buffer_log_interval: int = 10,
    glob_pattern: str = "fly*", 
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
    model_weights = config.ModelWeightsConfig(model_weights=model_checkpoint_path)
    model = BodySegmentationModel.create_architecture_from_config(
        architecture_config_path
    ).cuda()
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
        pred_dict = pipeline.inference(batch["frames"])
        logits = pred_dict["logits"]
        pred_seg = torch.argmax(logits, dim=1).to(torch.uint8).detach().cpu()
        confidence = (pred_dict["confidence"] * 100).to(torch.uint8).detach().cpu()

        for i in range(logits.shape[0]):
            data_item = (pred_seg[i, :, :], confidence[i, :, :])
            output_buffer.add_data(
                bucket=batch["video_indices"][i],
                index=batch["frame_indices"][i],
                data=data_item,
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
        help="Path to segment model directory. If not provided, will be loaded from config file.",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--epoch",
        type=int,
        help="Epoch number of the model checkpoint to use for inference.",
        default=14,
    )
    parser.add_argument(
        "--step",
        type=int,
        help="Step number of the model checkpoint to use for inference.",
        default=12000,
    )
    parser.add_argument(
        "--output_basedir",
        type=Path,
        help="Base directory to save output predictions. If not provided, will be saved in the model directory under production/epoch{epoch}_step{step}/",
        required=False,
        default=None,
    )

    args = parser.parse_args()
    
    return args.aligned_data_dir, args.glob_pattern, args.config_path, args.segment_model_dir, args.epoch, args.step, args.output_basedir

if __name__ == "__main__":
    # parse paths
    input_basedir, glob_pattern, config_path, segment_model_dir, epoch, step, output_basedir = start()
    if not segment_model_dir:
        # load from config file
        with open(config_path, "r") as f:
            prod_config = yaml.safe_load(f)
        model_dir = Path(prod_config["bodyseg"]["checkpoint"]).parent.parent
    else:
        model_dir = segment_model_dir

    batch_size = 192
    n_workers = 16
    inference_image_size = (256, 256)
    output_buffer_log_interval = 10
    epoch = 14  # chosen by validation performance and visual inspection
    step = 12000  # last step of each epoch

    model_checkpoint_path = model_dir / f"checkpoints/epoch{epoch}_step{step}.model.pth"
    if output_basedir is None:
        output_basedir = model_dir / f"production/epoch{epoch}_step{step}/"
    else: 
        output_basedir = output_basedir / f"bodyseg/epoch{epoch}_step{step}/"
    output_basedir.mkdir(parents=True, exist_ok=True)


    test_bodyseg_model(
        input_basedir=input_basedir,
        model_dir=model_dir,
        model_checkpoint_path=model_checkpoint_path,
        output_basedir=output_basedir,
        batch_size=batch_size,
        n_workers=n_workers,
        inference_image_size=inference_image_size,
        output_buffer_log_interval=output_buffer_log_interval,
        glob_pattern=glob_pattern,
    )
