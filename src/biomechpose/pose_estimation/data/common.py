import json
from pathlib import Path

from biomechpose.util import check_num_frames, read_frames_from_video


def get_video_metadata(
    sample_video_path: Path,
    cache_metadata: bool = True,
    use_cached_metadata: bool = True,
):
    cache_path = sample_video_path.with_suffix(".metadata.json")
    if use_cached_metadata and cache_path.is_file():
        try:
            with open(cache_path, "r") as f:
                metadata = json.load(f)
            n_frames = metadata["n_frames"]
            frame_size = tuple(metadata["frame_size"])
            fps = metadata["fps"]
        except Exception as e:
            print(f"Corrupted metadata cache file {cache_path}")
            raise e
    else:
        n_frames = check_num_frames(sample_video_path)
        sample_frames, fps = read_frames_from_video(
            sample_video_path, frame_indices=[0]
        )
        frame_size = sample_frames[0].shape[:2]

        if cache_metadata:
            metadata = {
                "n_frames": n_frames,
                "frame_size": list(frame_size),
                "fps": fps,
            }
            with open(cache_path, "w") as f:
                json.dump(metadata, f, indent=2)
    return {"n_frames": n_frames, "frame_size": frame_size, "fps": fps}
