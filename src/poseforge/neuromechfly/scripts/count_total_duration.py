from pathlib import Path
from tqdm import tqdm

from poseforge.util import check_num_frames


if __name__ == "__main__":
    simulation_results_basedir = Path("bulk_data/nmf_rendering")
    data_output_freq = 300  # Hz

    section_video_files = simulation_results_basedir.rglob(
        "segment_*/nmf_sim_render_colorcode_0.mp4"
    )
    section_video_files = list(sorted(section_video_files))
    print("Checking results before postprocessing...")
    total_frames = 0
    for file in tqdm(section_video_files, disable=None):
        total_frames += check_num_frames(file)
    total_seconds = total_frames / data_output_freq
    print(
        f"Before postprocessing: {total_frames} total frames ({total_seconds:.2f} sec)"
    )

    subsection_video_files = simulation_results_basedir.rglob(
        "subsegment_*/processed_nmf_sim_render_colorcode_0.mp4"
    )
    subsection_video_files = list(sorted(subsection_video_files))
    print("Checking results after postprocessing...")
    total_frames = 0
    for file in tqdm(subsection_video_files, disable=None):
        total_frames += check_num_frames(file)
    total_seconds = total_frames / data_output_freq
    print(
        f"After postprocessing: {total_frames} total frames ({total_seconds:.2f} sec)"
    )
