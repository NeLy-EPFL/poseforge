import torch
from pathlib import Path

from poseforge.pose.data.synthetic import AtomicBatchDataset


def test_atomic_batch_dataset_loading():
    data_dirs = [
        Path("bulk_data/pose_estimation/atomic_batches/BO_Gal4_fly1_trial001/")
    ]
    dataset = AtomicBatchDataset(
        data_dirs=data_dirs,
        n_variants=4,
        image_size=(256, 256),
        n_channels=1,
        frames_serialization_spacing=10,
        load_dof_angles=True,
        load_keypoint_positions=True,
        load_body_segment_maps=True,
    )
    print(f"Dataset length: {len(dataset)}")
    frames, labels = dataset[100]
    
    assert isinstance(frames, torch.Tensor)
    batch_size = 32
    assert frames.shape == (4, batch_size, 1, 256, 256)

    assert isinstance(labels, dict)
    assert "dof_angles" in labels
    assert "keypoint_pos" in labels
    assert "body_seg_maps" in labels
    assert labels["dof_angles"].shape == (batch_size, 42)
    assert labels["keypoint_pos"].shape == (batch_size, 32, 3)
    assert labels["body_seg_maps"].shape == (batch_size, 256, 256)