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
    video_path = subsegment_dir / "processed_simulation_rendering.mp4"
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
            input_batch = np.array(video_frames[i : i + batch_size])
            input_batch_pil = [to_pil_image(frame) for frame in input_batch]
            output_batch = inference_pipeline.infer(input_batch_pil)
            for j in range(output_batch.shape[0]):
                frame = output_batch[j]
                video_writer.append_data(frame)


if __name__ == "__main__":
    # Set data paths and configs
    trial_dirs = [
        Path("bulk_data/nmf_rendering/BO_Gal4_fly1_trial001/segment_000/subsegment_000")
    ]
    # checkpoint_path = Path(
    #     "bulk_data/style_transfer/checkpoints/test_trial_small/spotlight202506_to_aymanns2022/latest_net_G.pth"
    # )
    checkpoint_path = Path(
        "bulk_data/style_transfer/checkpoints/test_trial/spotlight202506_to_aymanns2022/latest_net_G.pth"
    )
    output_basedir = Path("bulk_data/style_transfer/synthetic_output")
    input_nc = 3
    output_nc = 3
    ngf = 64
    netG = "stylegan2"
    image_side_length = 256
    nce_layers = [0, 4, 8, 12, 16]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Set up inference pipeline (loading the model etc)
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

    for subsegment_dir in trial_dirs:
        trial, segment, subsegment = subsegment_dir.parts[-3:]
        output_path = output_basedir / f"{trial}_{segment}_{subsegment}.mp4"
        process_subsegment(inference_pipeline, subsegment_dir, output_path)
