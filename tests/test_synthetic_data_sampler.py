import pytest
import torch
import numpy as np
from pathlib import Path

from biomechpose.pose_estimation import SimulatedDataSequence, SyntheticFramesSampler


def test_sampling_contrastive_pretraining_dataset_sampling():
    video_basedir = Path("bulk_data/style_transfer/production/translated_videos/")
    simulations = [
        "BO_Gal4_fly1_trial001/segment_000/subsegment_000",
        "BO_Gal4_fly1_trial001/segment_000/subsegment_001",
    ]
    models = [
        "ngf16_netGsmallstylegan2_batsize2_lambGAN0.2_epoch121",
        "ngf16_netGstylegan2_batsize4_lambGAN0.2_epoch200",
    ]

    # Create SimulatedDataSequence objects for each simulation
    simulated_data_sequences = []
    for sim in simulations:
        video_paths = [
            video_basedir / sim / f"translated_{model}.mp4" for model in models
        ]
        sequence = SimulatedDataSequence(
            synthetic_video_paths=video_paths, sim_name=sim
        )
        simulated_data_sequences.append(sequence)

    batch_size = 4
    sampling_stride = 3
    dataset = SyntheticFramesSampler(
        simulated_data_sequences,
        batch_size=batch_size,
        sampling_stride=sampling_stride,
        transform=None,
        load_labels=False,
    )

    # Check dataset size
    sim_lengths = [len(seq) for seq in simulated_data_sequences]
    n_frames_total = sum(sim_lengths)
    super_block_size = batch_size * sampling_stride
    n_super_blocks = n_frames_total // super_block_size
    dataset_size = n_super_blocks * sampling_stride
    assert len(dataset) == dataset_size

    # Test sampling - let's try to sample values across two simulations
    num_full_super_blocks_0th_sim = sim_lengths[0] // super_block_size
    if sim_lengths[0] % super_block_size == 0:
        raise ValueError(
            "Unfortunate numbers chosen in test - cannot test edge case where a sample "
            "spans two simulations. This does not mean the code is wrong, just pick "
            "different numbers to fully test the code."
        )
    super_block_id = num_full_super_blocks_0th_sim
    offset = sampling_stride - 1
    sample_idx = super_block_id * sampling_stride + offset
    # This combo is guaranteed to sample frames from both simulations
    global_frame_ids, sim_ids, local_frame_ids = dataset.determine_batch_frame_ids(
        sample_idx
    )

    # Check global frame id sampling
    super_block_id = sample_idx // sampling_stride
    offset = sample_idx % sampling_stride
    first_global_frame_id = super_block_id * super_block_size + offset
    expected_global_frame_ids = [
        first_global_frame_id + i * sampling_stride for i in range(batch_size)
    ]
    assert global_frame_ids.tolist() == expected_global_frame_ids

    # Check sim ids
    assert (sim_ids == (global_frame_ids >= sim_lengths[0]).astype(int)).all()

    # Check local frame ids
    (local_frame_ids[sim_ids == 0] == global_frame_ids[sim_ids == 0]).all()
    (
        local_frame_ids[sim_ids == 1] == global_frame_ids[sim_ids == 1] - sim_lengths[0]
    ).all()


def test_sampling_contrastive_pretraining_dataset_data_loading():
    video_basedir = Path("bulk_data/style_transfer/production/translated_videos/")
    simulations = [
        "BO_Gal4_fly1_trial001/segment_000/subsegment_000",
        "BO_Gal4_fly1_trial001/segment_000/subsegment_001",
    ]
    models = [
        "ngf16_netGsmallstylegan2_batsize2_lambGAN0.2_epoch121",
        "ngf16_netGstylegan2_batsize4_lambGAN0.2_epoch200",
    ]

    # Create SimulatedDataSequence objects for each simulation
    simulated_data_sequences = []
    for sim in simulations:
        video_paths = [
            video_basedir / sim / f"translated_{model}.mp4" for model in models
        ]
        sequence = SimulatedDataSequence(
            synthetic_video_paths=video_paths, sim_name=sim
        )
        simulated_data_sequences.append(sequence)

    height, width = 256, 256
    batch_size = 4
    sampling_stride = 3
    dataset = SyntheticFramesSampler(
        simulated_data_sequences,
        batch_size=batch_size,
        sampling_stride=sampling_stride,
        transform=None,
        load_labels=False,
    )

    batch = dataset[0]
    assert batch.shape == (
        len(models),
        batch_size,
        1,
        height,
        width,
    )  # Note: 1 channel (grayscale)
    assert batch.dtype == torch.float32
    assert batch.min() >= 0.0 and batch.max() <= 1.0
    means = []
    for i in range(len(models)):
        for j in range(batch_size):
            means.append(batch[i, j].mean().item())
    means = np.array(means)
    assert means.mean() > 0.01 and means.mean() < 0.99
    assert len(set(means)) == len(means)  # All frames should be different


if __name__ == "__main__":
    pytest.main([__file__])
