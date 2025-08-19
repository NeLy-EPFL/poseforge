from pathlib import Path

from biomechpose.simulate_nmf.postprocessing import process_segment


if __name__ == "__main__":
    recording_dir = Path("bulk_data/nmf_rendering/BO_Gal4_fly1_trial001/segment_0")
    process_segment(
        recording_dir,
        start_frame=0,
        end_frame=-1,
        max_tilt_angle_deg=30.0,
        visualize=True,
        camera_elevation=30.0,
        max_abs_azimuth=30.0,
        azimuth_rotation_period=300.0,
    )
