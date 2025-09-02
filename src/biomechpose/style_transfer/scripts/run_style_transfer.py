import re
import numpy as np
import torch
import imageio.v2 as imageio
from pathlib import Path
from tqdm import trange
from torchvision.transforms.functional import to_pil_image

from biomechpose.style_transfer.cut_inference import InferencePipeline


def process_subsegment(
    inference_pipeline: InferencePipeline,
    subsegment_dir: Path,
    output_path: Path,
    batch_size: int = 4,
) -> None:
    video_path = subsegment_dir / "processed_nmf_sim_render_colorcode_0.mp4"
    with imageio.get_reader(str(video_path), "ffmpeg") as reader:
        fps = reader.get_meta_data()["fps"]
        video_frames = [frame for frame in reader]
    print(f"Read {len(video_frames)} frames from {video_path}")

    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create video writer
    with imageio.get_writer(
        str(output_path),
        "ffmpeg",
        fps=fps,
        codec="libx264",
        quality=10,  # 10 is highest for imageio, lower is lower quality
        ffmpeg_params=["-crf", "18", "-preset", "slow"],  # lower crf = higher quality
    ) as video_writer:
        # Process frames in batches
        for i in trange(0, len(video_frames), batch_size, disable=None):
            # Get batch of frames (each frame is HWC format, uint8, 0-255)
            input_batch_frames = video_frames[i : i + batch_size]
            
            # Convert each frame to PIL Image directly (imageio frames are already in the right format)
            from PIL import Image
            input_batch_pil = [Image.fromarray(frame) for frame in input_batch_frames]
            
            output_batch = inference_pipeline.infer(input_batch_pil)
            for j in range(output_batch.shape[0]):
                frame = output_batch[j]
                video_writer.append_data(frame)


def get_inference_pipeline(checkpoint_path: Path, params: dict) -> InferencePipeline:
    # Some parameters can be assumed
    input_nc = params.get("input_nc", 3)
    output_nc = params.get("output_nc", 3)
    image_side_length = params.get("image_side_length", 256)
    nce_layers = params.get("nce_layers", [0, 4, 8, 12, 16])

    # Other parameters must be stated explicitly
    ngf = params["ngf"]
    netG = params["net"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    inference_pipeline = InferencePipeline(
        checkpoint_path,
        input_nc=input_nc,
        output_nc=output_nc,
        ngf=ngf,
        netG=netG,
        image_side_length=image_side_length,
        nce_layers=nce_layers,
        device=device,
    )
    return inference_pipeline


if __name__ == "__main__":
    checkpoints_basedir = Path(
        "/home/sibwang/Data/scitas_data/biomechpose/bulk_data/style_transfer/20250828_parameter_sweep/checkpoints"
    )
    subsegment_dir = Path(
        "bulk_data/nmf_rendering/BO_Gal4_fly1_trial001/segment_007/subsegment_000"
    )

    # Detect trials and their parameters
    trial_name_regex = r"ngf(?P<ngf>\d+)_netG(?P<net>[a-zA-Z0-9]+)_batsize(?P<batsize>\d+)_lambGAN(?P<lambGAN>[\d.]+)"
    trials = sorted(list([x.name for x in checkpoints_basedir.iterdir() if x.is_dir()]))
    params_by_trial = {}
    for path in sorted(list(checkpoints_basedir.iterdir())):
        trial_name = path.name
        match = re.match(trial_name_regex, trial_name)
        if match:
            params_by_trial[trial_name] = {
                "ngf": int(match.group("ngf")),
                "net": match.group("net"),
                "batsize": int(match.group("batsize")),
                "lambGAN": float(match.group("lambGAN")),
            }
        else:
            print(f"Warning: Could not parse parameters from trial name: {trial_name}")

    # Test-run models one by one
    for trial_name, params in params_by_trial.items():
        print(f"=== Processing trial: {trial_name} with params: {params} ===")
        # Recursively search for "latest_net_G.pth" in the trial directory
        checkpoint_files = list(
            (checkpoints_basedir / trial_name).rglob("latest_net_G.pth")
        )
        if not checkpoint_files:
            print(
                f"Warning: No 'latest_net_G.pth' found in {checkpoints_basedir / trial_name}"
            )
            continue
        checkpoint_path = checkpoint_files[0]
        print(f"Using checkpoint: {checkpoint_path}")
        inference_pipeline = get_inference_pipeline(checkpoint_path, params)

        trial, segment, subsegment = subsegment_dir.parts[-3:]
        output_path = (
            Path("bulk_data/style_transfer/synthetic_output")
            / f"{trial_name}_{trial}_{segment}_{subsegment}.mp4"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        process_subsegment(inference_pipeline, subsegment_dir, output_path)
