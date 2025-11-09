"""
Utilities for video decoding operations using torchcodec.

torchcodec still looks pretty under-developed at this point. We are using v0.7.
The following functions are adapted from the tutorial at
https://meta-pytorch.org/torchcodec/0.7/generated_examples/decoding/parallel_decoding.html
"""

import torch
import logging
from pathlib import Path
from joblib import Parallel, delayed
from torchcodec.decoders import VideoDecoder


def split_indices(indices: list[int], num_chunks: int) -> list[list[int]]:
    """Split a list of indices into approximately equal chunks."""
    chunk_size = len(indices) // num_chunks
    chunks = []

    for i in range(num_chunks - 1):
        chunks.append(indices[i * chunk_size : (i + 1) * chunk_size])

    # Last chunk may be slightly larger
    chunks.append(indices[(num_chunks - 1) * chunk_size :])
    return chunks


def decode_sequentially(indices: list[int], video_path: Path):
    """Decode frames sequentially using a single decoder instance."""
    decoder = VideoDecoder(video_path, seek_mode="exact")
    return decoder.get_frames_at(indices)


def decode_with_multithreading(
    indices: list[int],
    num_threads: int,
    video_path: Path,
):
    """Decode frames using multiple threads with joblib."""
    chunks = split_indices(indices, num_chunks=num_threads)
    pool = Parallel(n_jobs=num_threads, prefer="threads", verbose=0)
    logger = logging.getLogger(__name__)
    logger.info(
        f"Decoding {len(indices)} frames from {video_path} using {num_threads} threads "
        f"(effectively {pool._effective_n_jobs()} threads)"
    )
    results = pool(delayed(decode_sequentially)(chunk, video_path) for chunk in chunks)
    return torch.cat([frame_batch.data for frame_batch in results], dim=0)
