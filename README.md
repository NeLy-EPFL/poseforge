# BioMechPose

Pose estimation guided by a biomechanical model.


## Complete pipeline and code structure
### Part I: Simulate motion priors in NeuroMechFly and generate renderings
1. Run `python src/biomechpose/simulate_nmf/scripts/copy_kinematic_recording.py`
    - This script scans data from Aymanns et al. (2022) from the NeLy lab server (also publicly available on Harvard Dataverse: https://doi.org/10.7910/DVN/QQMNQK), extracts key kinematic data, and saves them as pickle files.
2. Run `python src/biomechpose/simulate_nmf/scripts/run_simulation.py`
    - This script selects non-resting segments from the recorded kinematics from Aymanns et al. Then, it simulates the selected segments using NeuroMechFly (https://neuromechfly.org/) and saves the visual renderings.
    - Because Aymanns et al. reports _tethered_ fly behaviors, replaying them on flat terrain might result in failures (e.g. fly flipping upside down).  This script includes code that filters out such periods and further splits each segment into several (though typically just one) subsegments.

### Part II: Preprocess Spotlight behavior recordings
1. Run `python src/biomechpose/spotlight_pipeline/scripts/extract_spotlight_frames.py`
    - This script processes each Spotlight experimental trial by extracting, aligning, and cropping frames from the behavior video, and saving the processed frames as individual images in an output directory.
2. **[Only to train the flip detection model]** Run `python src/biomechpose/spotlight_pipeline/scripts/train_flip_detection_model.py`
    - This script trains a binary image classifier that detects whether the fly is flipped in the Spotlight arena.
    - Prerequisite: Manual labels of whether the fly is flipped must be supplied. This is done by creating a `manual_label/` subdirectory under the directory containing extracted frames from each Spotlight experimental trial, further creating a `manual_label/flipped` and a `manual_label/not_flipped` subdirectories, and copying the extracted frames into the appropriate folder.
3. Run `python src/biomechpose/spotlight_pipeline/scripts/detect_flipped_flies.py`
    - This script generates a label file indicating whether the fly is flipped in each extracted Spotlight behavioral frame. Those in which the fly is flipped will be excluded in subsequent steps.


### Part III: Translate NeuroMechFly renderings to the domain of Spotlight behavior recordings
1. Run `python src/biomechpose/style_transfer/scripts/extract_dataset.py`
    - This script randomly extracts subsets of NeuroMechFly rendering frames and Spotlight recording frames for training the style transfer model.
2. Run `bash src/biomechpose/style_transfer/scripts/train_cut_model_caller.sh`
    - This script trains a Contrastive Unpaired Translation (CUT) model using a demonstrative set of hyperparameters.
    - This shell script calls the CLI from `src/biomechpose/style_transfer/scripts/train_cut_model.py`. The advantage of having a shell script that calls a Python CLI is that we can change hyperparameters simply by passing them to the Python training script via the CLI from the shell script(s), as opposed to having to make multiple copies of the Python training script. This is handy for hyperparameter tuning.
    - Hyperparameters can be selected by training many models with different hyperparameters on a cluster (e.g. SCITAS). See `scripts_on_cluster/style_transfer_training/` for an example pipeline to machine-generate a batch of `*.run` scripts that can be submitted to the Slurm scheduler on a cluster.
    - In the training procedure, we use Weights and Biases (https://wandb.ai/) to simplify the task of monitoring the training runs and visualizing their results.

**[The following is only for evaluating the models and selecting the best one(s) for production]**

3. Run `python src/biomechpose/style_transfer/scripts/test_trained_models.py`
    - This script runs inference on a manually selected, representative set of simulation data, using checkpoints from different training stages of each training run (e.g. once every 20 epochs).
    - The user must manually specify a set of simulation data to use for testing and a set of model checkpoints to test. To do so, edit parameters in the `__main__` section of the script.
4. Run `python src/biomechpose/style_transfer/scripts/visualize_inference_results.py`
    - For each training run, this script merges videos its inference results at different stage of training into a single summary video for easier comparison. The original NeuroMechFly simulation rendering is also included in the summary video.
5. Manually generate a `bulk_data/style_transfer/synthetic_output/summary_videos/quality_assessment/human_annotated_scores.csv` file with the following columns:
    - `run`: Name of the training run, e.g. "ngf32_netGsmallstylegan2_batsize4_lambGAN0.1"
    - `best_epoch`: Epoch number of the best model in this run
    - `score`: Human-annotated score for the best model (1-5, higher is better)
    - `note`: Optional note about the run
6. Run `python src/biomechpose/style_transfer/scripts/visualize_human_annotated_scores.py`
    - This generates visualizations aimed to help refine model hyperparameters and iteratively retrain the models.


### Part IV: Generalized pose estimation
TODO