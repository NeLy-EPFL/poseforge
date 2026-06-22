import argparse
import logging
from pathlib import Path

import cv2
import torch
import numpy as np
from tqdm import tqdm
import yaml

from poseforge.pose.keypoints3d import Pose2p5DModel, Pose2p5DPipeline
from poseforge.pose.keypoints3d import config as keypoints3d_config

def main():
    parser = argparse.ArgumentParser(description="Run 3D keypoint inference on images and plot predictions.")
    parser.add_argument("input_path", type=Path, help="Path to the input image or directory of images.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to a checkpoint file under a model directory checkpoints folder.")
    parser.add_argument("--output_dir", type=Path, default=None, help="Directory to save the output images with plotted keypoints.")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for inference.")
    parser.add_argument("--n_images", type=int, default=None, help="Maximum number of images to process (for quick testing).")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for inference.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    input_path = args.input_path
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")
        
    if input_path.is_file():
        image_paths = [input_path]
    else:
        image_paths = []
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            image_paths.extend(list(input_path.glob(ext)))
        image_paths.sort()
        
    if not image_paths:
        raise ValueError(f"No images found in {input_path}")
        
    if args.n_images is not None:
        # randomly sample n_images if there are more than n_images
        if len(image_paths) > args.n_images:
            image_paths = np.random.choice(image_paths, size=args.n_images, replace=False).tolist()
        else:
            logging.warning(f"Requested n_images={args.n_images} but only found {len(image_paths)} images. Processing all images.")

    # Resolve model directory from checkpoint path.
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if checkpoint_path.parent.name != "checkpoints":
        raise ValueError(
            "Cannot infer model directory from checkpoint path. "
            "Expected checkpoint under '<model_dir>/checkpoints/*.model.pth', got: "
            f"{checkpoint_path}"
        )

    model_dir = checkpoint_path.parent.parent
    architecture_config_path = model_dir / "configs/model_architecture_config.yaml"
    if not architecture_config_path.is_file():
        architecture_config_path = model_dir / "model_architecture_config.yaml"
    if not architecture_config_path.is_file():
        raise FileNotFoundError(
            "Model architecture config not found. Tried: "
            f"{model_dir / 'configs/model_architecture_config.yaml'} and {model_dir / 'model_architecture_config.yaml'}"
        )

    # Load config to get expected image size
    data_config_path = model_dir / "configs/data_config.yaml"
    if not data_config_path.is_file():
        data_config_path = model_dir / "data_config.yaml"
    if not data_config_path.is_file():
        raise FileNotFoundError(
            "Data config not found. Tried: "
            f"{model_dir / 'configs/data_config.yaml'} and {model_dir / 'data_config.yaml'}"
        )
    
    with open(data_config_path, "r") as f:
        data_config = yaml.safe_load(f)
    image_size = tuple(data_config.get("input_image_size", (256, 256)))
    
    logging.info(f"Processing {len(image_paths)} images")
    logging.info(f"Using image size: {image_size}")
    logging.info(f"Loading checkpoint: {checkpoint_path}")
    logging.info(f"Inferred model directory: {model_dir}")

    # Build model and pipeline
    architecture_config = keypoints3d_config.ModelArchitectureConfig.load(architecture_config_path)
    model = Pose2p5DModel.create_architecture_from_config(architecture_config)
    model.load_weights_from_config(keypoints3d_config.ModelWeightsConfig(model_weights=str(checkpoint_path)))
    pipeline = Pose2p5DPipeline(model, device=args.device, use_float16=True)

    # Prepare output dir
    output_dir = args.output_dir
    if output_dir is None:
        pred_folder = f"{model_dir.name}_{checkpoint_path.name.split(".")[0]}_predictions"
        if input_path.is_file():
            output_dir = input_path.parent / pred_folder
        else:
            output_dir = input_path / pred_folder
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    def process_batch(paths, frames, orig_frames):
        batch_tensor = np.zeros((len(frames), 3, image_size[0], image_size[1]), dtype=np.float32)
        for i, f in enumerate(frames):
            f_rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            f_resized = cv2.resize(f_rgb, (image_size[1], image_size[0]))
            batch_tensor[i] = f_resized.transpose(2, 0, 1) / 255.0
            orig_frames[i] = cv2.resize(f, (image_size[1], image_size[0]))
            
        tensor = torch.from_numpy(batch_tensor)
        pred_dict = pipeline.inference(tensor)
        pred_xy = pred_dict["pred_xy"].detach().cpu().numpy()
        
        for i in range(len(frames)):
            img = orig_frames[i]
            pts = pred_xy[i]
            for pt in pts:
                x, y = int(pt[0]), int(pt[1])
                cv2.circle(img, (x, y), radius=2, color=(0, 0, 255), thickness=-1)
            
            out_path = output_dir / f"{paths[i].stem}_pred.png"
            cv2.imwrite(str(out_path), img)

    paths_buffer = []
    frames_buffer = []
    orig_frames_buffer = []
    
    pbar = tqdm(total=len(image_paths))
    
    for img_path in image_paths:
        frame = cv2.imread(str(img_path))
        if frame is None:
            logging.warning(f"Could not read image: {img_path}")
            pbar.update(1)
            continue
            
        paths_buffer.append(img_path)
        frames_buffer.append(frame)
        orig_frames_buffer.append(frame.copy())
        
        if len(frames_buffer) == args.batch_size:
            process_batch(paths_buffer, frames_buffer, orig_frames_buffer)
            paths_buffer = []
            frames_buffer = []
            orig_frames_buffer = []
            
        pbar.update(1)

    if len(frames_buffer) > 0:
        process_batch(paths_buffer, frames_buffer, orig_frames_buffer)
        
    pbar.close()
    
    logging.info(f"Finished processing. Outputs saved to {output_dir}")

if __name__ == "__main__":
    main()
