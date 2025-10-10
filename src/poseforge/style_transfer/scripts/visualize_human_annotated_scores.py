"""This script requires a human-annotated score file, which is a CSV file
with the following columns:
- run: Name of the training run, e.g. "ngf32_netGsmallstylegan2_batsize4_lambGAN0.1"
- best_epoch: Epoch number of the best model in this run
- score: Human-annotated score for the best model (1-5, higher is better)
- note: Optional note about the run

Pipeline:
1. Run test_trained_models.py to run inference on a manually selected
   representative set of simulation data using checkpoints from different
   training stages of different training runs.
2. Run visualize_inference_results.py to generate summary videos of
   inference results in each training run at different training stages.
3. Manually evaluate the summary videos and record scores in a CSV file.
4. Run this script to parse the scores and visualize the relationship
   between hyperparameters and scores.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

from poseforge.style_transfer import parse_hyperparameters_from_trial_name


def expand_human_generated_scores_file(scores_df_raw: pd.DataFrame) -> pd.DataFrame:
    """Given a DataFrame containing human evaluation scores, expand it to
    include hyperparameter information parsed from the run names."""
    columns = defaultdict(list)
    for _, entry in scores_df_raw.iterrows():
        run_name = entry["run"]
        hparams = parse_hyperparameters_from_trial_name(run_name)
        if "-cont" in run_name:
            continuation_idx = int(run_name.split("-")[1].replace("cont", ""))
        else:
            continuation_idx = 0
        if hparams["net"] == "smallstylegan2":
            num_blocks = 2
        elif hparams["net"] == "stylegan2":
            num_blocks = 6
        else:
            raise ValueError(f"Unknown net type: {hparams['net']}")
        columns["ngf"].append(hparams["ngf"])
        columns["net"].append(hparams["net"])
        columns["num_blocks"].append(num_blocks)
        columns["batsize"].append(hparams["batsize"])
        columns["lambGAN"].append(hparams["lambGAN"])
        columns["run"].append(run_name)
        columns["continuation_idx"].append(continuation_idx)
        columns["best_epoch"].append(entry["best_epoch"])
        columns["best_score"].append(entry["score"])
        columns["note"].append("" if pd.isna(entry["note"]) else entry["note"])
    return pd.DataFrame(data=columns)


if __name__ == "__main__":
    # Configurations
    qa_dir = Path(
        "bulk_data/style_transfer/synthetic_output/summary_videos/quality_assessment"
    )
    human_eval_scores_path = qa_dir / "human_annotated_scores.csv"
    visualizations_output_dir = qa_dir
    hyperparameters = ["ngf", "num_blocks", "batsize", "lambGAN"]

    visualizations_output_dir.mkdir(parents=True, exist_ok=True)

    # Load and parse human evaluation scores
    scores_df_raw = pd.read_csv(human_eval_scores_path)
    run_info_df = expand_human_generated_scores_file(scores_df_raw)

    # Plot pairwise relationships
    sns.pairplot(
        run_info_df,
        vars=hyperparameters,
        hue="best_score",
        palette="viridis",
        hue_order=None,
        diag_kind="kde",
    )
    plt.savefig(visualizations_output_dir / "pairplot.png")
    plt.close()

    # Plot best scores vs. values for each hyperparameter
    fig, axes = plt.subplots(
        1,
        len(hyperparameters),
        figsize=(3 * len(hyperparameters), 3),
        tight_layout=True,
    )
    for ax, param in zip(axes, hyperparameters):
        x = run_info_df[param]
        y = run_info_df["best_score"]
        # Add small random noise to x and y to help visualize overlapping dots
        x_noise = np.random.normal(0, 0.02 * (x.max() - x.min()), size=len(run_info_df))
        y_noise = np.random.normal(0, 0.02 * (y.max() - y.min()), size=len(run_info_df))
        jitter_x = run_info_df[param] + x_noise
        jitter_y = run_info_df["best_score"] + y_noise
        ax.scatter(jitter_x, jitter_y, s=3, marker=".")
        ax.set_xlabel(param)
        ax.set_ylabel("Best score")
    plt.savefig(visualizations_output_dir / "scatterplot.png")
    plt.close()
