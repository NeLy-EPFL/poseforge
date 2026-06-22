import argparse
import h5py
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def get_colors(num_classes):
    cmap = plt.get_cmap('gist_rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, num_classes)]
    # Convert RGBA (0-1) to BGR (0-255) for OpenCV
    bgr_colors = [(int(c[2]*255), int(c[1]*255), int(c[0]*255)) for c in colors]
    bgr_colors[0] = (0, 0, 0) # Background is black/transparent
    return bgr_colors

def overlay_segmentation(frame, segmap, colors, alpha=0.5):
    # Create colored image from segmentation map
    colored_seg = np.zeros_like(frame)
    for c in range(1, len(colors)): # Skip background
        mask = segmap == c
        colored_seg[mask] = colors[c]
    
    # Overlay using alpha blending where mask is > 0
    fg_mask = segmap > 0
    overlay = frame.copy()
    overlay[fg_mask] = cv2.addWeighted(frame[fg_mask], 1 - alpha, colored_seg[fg_mask], alpha, 0)
    return overlay

def main():
    parser = argparse.ArgumentParser(description="Overlay body segmentation predictions on video.")
    parser.add_argument("h5_path", type=Path, help="Path to the .h5 prediction file.")
    parser.add_argument("video_path", type=Path, help="Path to the matching input video file (.mp4).")
    parser.add_argument("--alpha", type=float, default=0.5, help="Overlay transparency.")
    parser.add_argument("--dataset_name", type=str, default="pred_segmap", help="Name of the HDF5 dataset to overlay (e.g. pred_segmap, pred_segmap_raw)")
    
    args = parser.parse_args()
    
    if not args.h5_path.exists():
        raise FileNotFoundError(f"H5 file not found: {args.h5_path}")
    if not args.video_path.exists():
        raise FileNotFoundError(f"Video file not found: {args.video_path}")
        
    # Generate output filename in the same folder as H5 file
    video_stem = args.video_path.stem
    h5_stem = args.h5_path.stem
    out_path = args.h5_path.parent / f"{video_stem}_{h5_stem}_overlay.mp4"
    
    print(f"Reading predictions from {args.h5_path} (dataset: {args.dataset_name})")
    with h5py.File(args.h5_path, "r") as f:
        if args.dataset_name not in f:
            raise KeyError(f"Dataset '{args.dataset_name}' not found in H5 file. Available datasets: {list(f.keys())}")
        
        segmaps = f[args.dataset_name][:]
        num_classes = len(f[args.dataset_name].attrs.get("class_labels", []))
        if num_classes == 0:
            num_classes = segmaps.max() + 1
            
        frame_ids = f["frame_ids"][:] if "frame_ids" in f else None

    print(f"Loaded {len(segmaps)} segmentation frames.")
    colors = get_colors(num_classes)
    
    cap = cv2.VideoCapture(str(args.video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file {args.video_path}")
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    
    print(f"Writing overlay to {out_path}")
    
    pred_idx = 0
    frame_idx = 0
    
    # We might need to resize the segmap to match the video, as inference uses (256, 256)
    seg_h, seg_w = segmaps.shape[1], segmaps.shape[2]
    needs_resize = (seg_w != width) or (seg_h != height)
    
    with tqdm(total=len(segmaps)) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # If we have frame_ids, we only overlay on frames that have predictions
            has_pred = False
            if frame_ids is not None:
                if pred_idx < len(frame_ids) and frame_ids[pred_idx] == frame_idx:
                    has_pred = True
            else:
                # If no frame_ids, assume 1:1 mapping sequentially
                if pred_idx < len(segmaps):
                    has_pred = True
                    
            if has_pred:
                segmap = segmaps[pred_idx]
                if needs_resize:
                    # Nearest neighbor interpolation to preserve class labels
                    segmap = cv2.resize(segmap, (width, height), interpolation=cv2.INTER_NEAREST)
                    
                frame = overlay_segmentation(frame, segmap, colors, args.alpha)
                pred_idx += 1
                pbar.update(1)
                
            out.write(frame)
            frame_idx += 1

    cap.release()
    out.release()
    print(f"Done! Saved to {out_path}")

if __name__ == "__main__":
    main()
