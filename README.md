# BioMechPose

Pose estimation guided by a biomechanical model.

## Part I: Contrastive Unpaired Translation (CUT) for Spotlight image translation
### Pipeline
1. `python src/biomechpose/simulate_nmf/scripts/copy_kinematic_recording.py`
2. `python src/biomechpose/simulate_nmf/scripts/run_simulation.py`
3. `python src/biomechpose/spotlight_pipeline/scripts/extract_spotlight_frames.py`
4. `python src/biomechpose/spotlight_pipeline/scripts/train_flip_detection_model.py` (run only once)
5. `python src/biomechpose/spotlight_pipeline/scripts/detect_flipped_flies.py`


## Part II: Transformer for continuous pose estimation
TODO