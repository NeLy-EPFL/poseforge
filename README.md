# Body-aware enhanced pose estimation for _Drosophila melanogaster_

Pose estimation guided by a biomechanical model.

## Installation
This package used Poetry for package management. First, [install Poetry](https://python-poetry.org/docs/#installation) and optionally create a virtual environment (either through Poetry or through Conda). Then,
```bash
git clone git@github.com:NeLy-EPFL/poseforge.git
cd poseforge
poetry install
```

> [!IMPORTANT]
> This package requires [Sibo's fork](https://github.com/sibocw/contrastive-unpaired-translation) of [taesungp/contrastive-unpaired-translation](https://github.com/taesungp/contrastive-unpaired-translation) for Contrastive Unpaired Translation [(Park et al., 2020)](https://taesung.me/ContrastiveUnpairedTranslation/) and Sibo's [parallel-video-io](https://github.com/sibocw/parallel-video-io) package.
>
> These dependencies are already specified from in `pyproject.toml` via Github URLs. When you run the command above, these should be installed automatically. However, if these projects get updates pushed to Github, you might need to run `poetry update contrastive-unpaired-translation` and `poetry update parallel-video-io` manually.
>
> Alternatively, you can install the packages above in "edit mode" to gain better control:
>
> ```bash
> git clone https://github.com/sibocw/contrastive-unpaired-translation.git
> cd contrastive-unpaired-translation
> pip install -e .
> cd ..
> git clone https://github.com/sibocw/parallel-video-io.git
> cd parallel-video-io
> pip install -e .
> ```
> (Do not try to run `pip install -e package-name` directly without `cd`ing into `package-name`. The current directory matters with `pip install -e`.)
>
> **Important:** This package also depends on spotlight-tools from [spotlight-control](https://github.com/NeLy-EPFL/spotlight-control) (TODO).
> **Important:** This package also depends on spotlight-tools from [SeqIKPy](https://github.com/NeLy-EPFL/sequential-inverse-kinematics) (TODO).

## Complete pipeline and code structure
### Part I: Simulate motion priors in NeuroMechFly and generate renderings
1. Run `python src/poseforge/neuromechfly/scripts/copy_kinematic_recording.py`
    - This script scans data from Aymanns et al. (2022) from the NeLy lab server (also publicly available on Harvard Dataverse: https://doi.org/10.7910/DVN/QQMNQK), extracts key kinematic data, and saves them as pickle files.
2. Run `python src/poseforge/neuromechfly/scripts/run_simulation.py`
    - This script selects non-resting segments from the recorded kinematics from Aymanns et al. Then, it simulates the selected segments using [NeuroMechFly](https://neuromechfly.org/) and saves the visual renderings.
    - Because Aymanns et al. reports _tethered_ fly behaviors, replaying them on flat terrain might result in failures (e.g. fly flipping upside down).  This script includes code that filters out such periods and further splits each segment into several (though typically just one) subsegments.

### Part II: Preprocess Spotlight behavior recordings
1. Run `python src/poseforge/spotlight/scripts/extract_spotlight_frames.py`
    - This script processes each Spotlight experimental trial by extracting, aligning, and cropping frames from the behavior video, and saving the processed frames as individual images in an output directory.

> [!NOTE]
> The following step is only for training the flip detection model. It should not be used during production.
>
> 2. Run `python src/poseforge/spotlight/scripts/train_flip_detection_model.py`
    - This script trains a binary image classifier that detects whether the fly is flipped in the Spotlight arena.
    - Prerequisite: Manual labels of whether the fly is flipped must be supplied. This is done by creating a `manual_label/` subdirectory under the directory containing extracted frames from each Spotlight experimental trial, further creating a `manual_label/flipped` and a `manual_label/not_flipped` subdirectories, and copying the extracted frames into the appropriate folder.

3. Run `python src/poseforge/spotlight/scripts/detect_flipped_flies.py`
    - This script generates a label file indicating whether the fly is flipped in each extracted Spotlight behavioral frame. Those in which the fly is flipped will be excluded in subsequent steps.


### Part III: Translate NeuroMechFly renderings to the domain of Spotlight behavior recordings
1. Run `python src/poseforge/style_transfer/scripts/extract_dataset.py`
    - This script randomly extracts subsets of NeuroMechFly rendering frames and Spotlight recording frames for training the style transfer model.
2. Run `bash src/poseforge/style_transfer/scripts/train_cut_model_caller.sh`
    - This script trains a [Contrastive Unpaired Translation (CUT)](https://taesung.me/ContrastiveUnpairedTranslation/) model using a demonstrative set of hyperparameters.
    - This shell script calls the CLI from `src/poseforge/style_transfer/scripts/train_cut_model.py`. The advantage of having a shell script that calls a Python CLI is that we can change hyperparameters simply by passing them to the Python training script via the CLI from the shell script(s), as opposed to having to make multiple copies of the Python training script. This is handy for hyperparameter tuning.
    - Hyperparameters can be selected by training many models with different hyperparameters on a cluster (e.g. [SCITAS](https://www.epfl.ch/research/facilities/scitas/)). See `scripts_on_cluster/style_transfer_training/` for an example pipeline to machine-generate a batch of `*.run` scripts that can be submitted to the [Slurm](https://slurm.schedmd.com/documentation.html) scheduler on a cluster.
    - In the training procedure, we use [Weights and Biases](https://wandb.ai/) to simplify the task of monitoring the training runs and visualizing their results.

> [!NOTE]
> The following steps are only for evaluating trained models and selecting the best one(s). They should not be used during inference time.
> 
> 3. Run `python src/poseforge/style_transfer/scripts/test_trained_models.py`
>     - This script runs inference on a manually selected, representative set of simulation data, using checkpoints from different training stages of each training run (e.g. once every 20 epochs).
>     - The user must manually specify a set of simulation data to use for testing and a set of model checkpoints to test. To do so, edit parameters in the `__main__` section of the script.
> 4. Run `python src/poseforge/style_transfer/scripts/visualize_inference_results.py`
>     - For each training run, this script merges videos its inference results at different stage of training into a single summary video for easier comparison. The original NeuroMechFly simulation rendering is also included in the summary video.
> 5. Manually generate a `bulk_data/style_transfer/synthetic_output/summary_videos/quality_assessment/human_annotated_scores.csv` file with the following columns:
>     - `run`: Name of the training run, e.g. "ngf32_netGsmallstylegan2_batsize4_lambGAN0.1"
>     - `best_epoch`: Epoch number of the best model in this run
>     - `score`: Human-annotated score for the best model (1-5, higher is better)
>     - `note`: Optional note about the run
> 6. Run `python src/poseforge/style_transfer/scripts/visualize_human_annotated_scores.py`
>     - This generates visualizations aimed to help refine model hyperparameters and iteratively retrain the models.

7. Run `python src/poseforge/style_transfer/scripts/run_inference.py`
    - This script uses a selected trained style transfer model to translate all NeuroMechFly rendering data into the domain of Spotlight behavior recordings.


### Part IV: Contrastive pretraining on synthetic data
1. Pre-shuffle the synthetic (and experimental) dataset using `python src/poseforge/pose/contrast/scripts/preextract_atomic_batches.py`. This will save small "atomic batches" of data that are always used together during training.
    - The Python file above is a CLI (run it with `-h` to see the help message). An example call of the CLI is included in the `__main__` section of the script. Alternatively, one can import the `extract_atomic_batches` function from this file and use it natively within Python (an example is included in the `__main__` section).
    - To run this on the SCITAS cluster (Jed), see `scripts_on_cluster/atomic_batch_extraction`.

> [!NOTE]
> The following step are only for pretraining the feature extractor with contrastive pretraining. It does not need to be rerun during production.
> 
> 2. Pretrain a ResNet18 feature extractor using `python src/poseforge/pose/contrast/scripts/run_contrastive_pretraining.py`.
>     - The Python file above is a CLI (run it with `-h` to see the help message). An example call of the CLI is included in the `__main__` section of the script. Alternatively, invoke training natively within Python by uncommenting example code in the `__main__` section.
>     - To train the model on the SCITAS cluster (Kuma), see `scripts_on_cluster/contrastive_pretraining_training`

> [!NOTE]
> The following steps are only for sanity-checking and visualizing the outcome of the constrastive pretraining step above. They do not need to be rerun during production. In inference time, the feature extractor will be used as a part of the pose estimation model.
>
> 3. Run inference using `python src/poseforge/pose/contrast/scripts/run_feature_extractor_inference.py`.
>      - The Python file above is a CLI (run it with `-h` to see the help message). An example call of the CLI is included in the `__main__` section of the script. Alternatively, invoke inference natively within Python by uncommenting example code in the `__main__` section.
>      - To run inference on the SCITAS cluster (Kuma), see `scripts_on_cluster/contrastive_pretraining_inference`. Note that running inference on all data will produce 200–300 GB of data. For quick inspection, it probably suffice to run inference only for one trial, one fly (e.g. `fly5_trial005` reserved for testing).
> 4. Run `python src/poseforge/pose/contrast/scripts/visualize_latents.py` to generate videos showing the latent-space trajectories of selected behavior snippets.

### Part V: Generalized pose estimation
#### Predicting 3D keypoint positions
> [!NOTE]
> The following steps are only for training the model and visualizing its performance on synthetic data. They do not need to be rerun during production.
>
> 1. Train 3D keypoint detection model using `python src/poseforge/pose/keypoints3d/scripts/run_keypoints3d_training.py`.
>     - This is a CLI (run `python run_keypoints3d_training.py -h` to see usage). However, the `__main__` section of this script also includes a commented-out example of how to run training directly within Python.
>     - See `scripts_on_cluster/keypoints3d_training/` for running on the SCITAS cluster. 
> 2. Visualize the performance of the model on synthetic data using `python src/poseforge/pose/keypoints3d/scripts/test_keypoints3d_models.py`. Note that you must select a particular model checkpoint file, and it doesn't necessarily have to be final model after the last epoch (observe validation loss to help decide which epoch to use).

3. Run inference on Spotlight data by running `python src/poseforge/pose/keypoints3d/scripts/run_keypoints3d_inference.py`. This script actually runs prediction using the model state at the end of every other epoch. Combined with the next step, this is meant to help select the best checkpoint to use in production.
4. Optionally, if you wish to visualize the output of the 3D keypoint detection model, run `python src/poseforge/pose/keypoints3d/scripts/visualize_production_keypoints3d.py`. Use the output to decide which checkpoint to use for production-time inference.
5. Run inverse kinematics by running `python src/poseforge/pose/keypoints3d/scripts/run_inverse_kinematics.py`. After inferring joint angles via IK, this script also runs forward kinematics to determine a new set of body-size-constrained 3D keypoint positions.

#### Predicting 2D body segmentation map
> [!NOTE]
> The following step is only for training the model; it does not have to be rerun in production.
>
> 1. Train the model: `python src/poseforge/pose/bodyseg/scripts/run_bodyseg_training.py`
>     - See `scripts_on_cluster/bodyseg_training` for running on the SCITAS cluster.

2. Run inference using trained mode: `python src/poseforge/pose/bodyseg/scripts/run_bodyseg_inference.py`
    - Note: you must first select a checkpoint to use (for example, by inspecting the logs). Specify the checkpoint in the `if __name__ == "__main__"` section of this script.
3. Optionally, visualize the results using `python src/poseforge/pose/bodyseg/scripts/visualize_bodyseg_predictions.py`. Similarly, you must specify a checkpoint to use. See the end of this script.