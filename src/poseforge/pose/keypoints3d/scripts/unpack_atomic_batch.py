import argparse
from pathlib import Path
import cv2
import numpy as np

from poseforge.pose.data.synthetic.atomic_batch import AtomicBatchDataset

def main():
    parser = argparse.ArgumentParser(description="Unpack an atomic batch video into single images.")
    parser.add_argument("video_path", type=Path, help="Path to the atomic batch mp4 file.")
    parser.add_argument("--output_dir", type=Path, default=None, help="Directory to save the extracted images.")
    parser.add_argument("--n_variants", type=int, default=4, help="Number of variants in the atomic batch.")
    parser.add_argument("--image_size", type=int, nargs=2, default=(256, 256), help="Image size (height width).")
    parser.add_argument("--spacing", type=int, default=10, help="Spacing between variants in the atomic batch.")
    parser.add_argument("--n_channels", type=int, default=3, help="Number of channels (usually 1 or 3).")
    args = parser.parse_args()

    video_path = args.video_path
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = video_path.parent / f"{video_path.stem}_unpacked"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading atomic batch from {video_path}...")
    frames_tensor = AtomicBatchDataset.load_atomic_batch_frames(
        video_path=video_path,
        n_variants=args.n_variants,
        image_size=tuple(args.image_size),
        n_channels=args.n_channels,
        spacing=args.spacing
    )
    
    # frames_tensor has shape (n_variants, n_frames, n_channels, height, width)
    # values are float32 in [0, 1]
    n_variants, n_frames, n_channels, height, width = frames_tensor.shape
    print(f"Unpacking {n_variants} variants and {n_frames} frames...")

    frames_np = frames_tensor.numpy()

    for v in range(n_variants):
        for f in range(n_frames):
            # shape (C, H, W)
            img = frames_np[v, f]
            # transpose to (H, W, C)
            img = img.transpose(1, 2, 0)
            # scale to [0, 255] uint8
            img = (img * 255).astype(np.uint8)
            
            if n_channels == 3:
                # convert RGB to BGR for cv2
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            out_filename = output_dir / f"variant{v:02d}_frame{f:04d}.png"
            cv2.imwrite(str(out_filename), img)

    print(f"Saved {n_variants * n_frames} images to {output_dir}")

if __name__ == "__main__":
    main()
