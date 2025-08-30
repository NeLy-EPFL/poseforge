from pathlib import Path

from biomechpose.simulate_nmf.data import load_kinematic_recording
from biomechpose.simulate_nmf.simulate import simulate_one_segment
from biomechpose.simulate_nmf.postprocessing import postprocess_segment


if __name__ == "__main__":
    kinematic_recording_dir = Path("bulk_data/kinematic_prior/aymanns2022/trials/")
    output_dir = Path("bulk_data/nmf_rendering_enhanced")
    input_timestep = 0.01
    sim_timestep = 0.0001
    max_segments = 1000  # limit to this many segments per trial

    # trial_paths = sorted(list(kinematic_recording_dir.glob("*.pkl")))
    trial_paths = [
        Path("bulk_data/kinematic_prior/aymanns2022/trials/BO_Gal4_fly1_trial002.pkl")
    ]

    for trial_path in trial_paths:
        trial_name = trial_path.stem
        kinematic_recording_segments = load_kinematic_recording(
            recording_path=trial_path,
            min_duration_frames=10,
            filter_size=5,
            filtered_frac_threshold=0.5,
        )
        kinematic_recording_segments = kinematic_recording_segments[:max_segments]
        num_segments = len(kinematic_recording_segments)
        print(f"### Processing trial: {trial_name} ({num_segments} segments) ###")
        for segment_id, segment in enumerate(kinematic_recording_segments):
            print(f"=== Simulating segment {segment_id + 1}/{num_segments} ===")
            output_subdir = output_dir / trial_name / f"segment_{segment_id:03d}"
            segment = segment.iloc[:20]
            # simulate_one_segment(segment, output_subdir, input_timestep, sim_timestep)
            postprocess_segment(output_subdir, visualize=True)
            break
        break
