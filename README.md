# BioMechPose

Pose estimation guided by a biomechanical model.

## Part I: CycleGAN for Fruit Fly Image Translation

This implementation provides a complete CycleGAN pipeline for translating between simulated (RGB) and experimental (grayscale) fruit fly images for pose estimation training data generation.

### Features

- **Domain Translation**: RGB simulated ↔ Grayscale experimental fruit fly images
- **Configurable Hyperparameters**: All key hyperparameters are exposed and documented
- **Modern Training**: Includes image buffers, proper initialization, and learning rate scheduling
- **Comprehensive Logging**: TensorBoard and Weights & Biases support
- **Visualization**: Training progress visualization and inference comparisons
- **Checkpointing**: Automatic checkpoint saving and resuming
- **Inference Pipeline**: Easy-to-use inference script for batch or single image translation

### Quick Start

### 1. Prepare Your Data

Organize your images in two directories:
```
data/
├── simulated/          # RGB images (3 channels)
│   ├── sim_001.png
│   ├── sim_002.png
│   └── ...
└── experimental/       # Grayscale images (1 channel)
    ├── exp_001.png
    ├── exp_002.png
    └── ...
```

#### 2. Train the Model

```bash
# Basic training
python train.py --sim_images_dir data/simulated --exp_images_dir data/experimental

# With Weights & Biases logging
python train.py --sim_images_dir data/simulated --exp_images_dir data/experimental --use_wandb

# Custom hyperparameters
python train.py \
    --sim_images_dir data/simulated \
    --exp_images_dir data/experimental \
    --batch_size 2 \
    --num_epochs 150 \
    --lambda_cycle 15.0 \
    --lr_generator 0.0001
```

#### 3. Run Inference

```bash
# Single image translation (simulated to experimental)
python infer.py \
    --model_path outputs/models/best_model.pth \
    --input_image test_simulated.png \
    --direction sim_to_exp \
    --output_dir results/

# Batch translation (experimental to simulated)
python infer.py \
    --model_path outputs/models/best_model.pth \
    --input_dir test_images/ \
    --direction exp_to_sim \
    --output_dir results/
```

### Key Hyperparameters

#### Model Architecture
- `generator_base_filters`: Base filters in generator (default: 64)
- `generator_n_residual_blocks`: ResNet blocks (9 for 512+, 6 for 256, default: 9)
- `discriminator_n_layers`: Discriminator layers (default: 3)

#### Training
- `batch_size`: Batch size (CycleGAN typically uses 1, default: 1)
- `num_epochs`: Training epochs (default: 200)
- `lr_generator`/`lr_discriminator`: Learning rates (default: 0.0002)
- `lambda_cycle`: Cycle consistency weight (default: 10.0)
- `lambda_identity`: Identity loss weight (default: 0.5, set to 0 to disable)

#### Data Augmentation
- `horizontal_flip_prob`: Probability of horizontal flip (default: 0.5)
- `brightness_jitter`/`contrast_jitter`: Color jittering (default: 0.1)
- `random_crop_size`: Random crop size (None to disable)

### File Structure

```
cyclegan_fruit_fly/
├── model.py              # CycleGAN model architecture
├── dataset.py            # Dataset and data loading utilities
├── visualization.py      # Visualization and logging functions
├── utils.py              # Utility functions (checkpointing, etc.)
├── train.py              # Training script
├── infer.py              # Inference script
└── README.md             # This file
```

### Training Output Structure

```
outputs/
├── checkpoints/          # Training checkpoints
│   ├── checkpoint_epoch_10.pth
│   └── ...
├── models/               # Best/final models
│   ├── best_model.pth
│   └── final_model.pth
├── visualizations/       # Training visualizations
│   ├── epoch_0_batch_0.png
│   └── ...
└── logs/                 # TensorBoard logs
    └── events.out.tfevents...
```

### Model Architecture Details

#### Generators
- **Input**: RGB (3-channel) or Grayscale (1-channel) 900×900 images
- **Architecture**: Encoder-ResNet-Decoder with reflection padding
- **ResNet Blocks**: 9 blocks for high-resolution images
- **Output**: Translated images with Tanh activation ([-1, 1] range)

#### Discriminators
- **Architecture**: PatchGAN discriminator
- **Receptive Field**: 70×70 patches
- **Layers**: 3 convolutional layers with InstanceNorm

#### Loss Functions
- **Adversarial Loss**: MSE loss for stable training
- **Cycle Consistency Loss**: L1 loss between reconstructed and original images
- **Identity Loss**: L1 loss to preserve color composition (optional)

### Training Tips

1. **Batch Size**: Start with batch_size=1. Increase if you have sufficient GPU memory.

2. **Cycle Consistency Weight**: `lambda_cycle=10.0` works well. Increase for stronger structural preservation.

3. **Identity Loss**: Set `lambda_identity=0.5` for natural images, or 0 to disable.

4. **Learning Rate Decay**: Starts after epoch 100 by default. Adjust `lr_decay_start_epoch`.

5. **Data Balance**: The dataset uses the longer dataset's length and cycles through the shorter one.

6. **GPU Memory**: Use `batch_size=1` for 900×900 images. Reduce `image_size` if needed.

### Monitoring Training

#### TensorBoard
```bash
tensorboard --logdir outputs/logs
```

#### Weights & Biases
Add `--use_wandb` flag to training command. Set project name with `--wandb_project`.

### Common Issues and Solutions

#### Out of Memory
- Reduce `batch_size` to 1
- Reduce `image_size`
- Reduce `generator_n_residual_blocks` to 6

#### Training Instability
- Ensure proper data normalization (handled automatically)
- Check that both domains have sufficient diversity
- Consider reducing learning rates

#### Poor Translation Quality
- Increase `lambda_cycle` for better structural preservation
- Ensure data domains are reasonably similar
- Increase training epochs

#### Color Artifacts
- Use `lambda_identity > 0` to preserve color composition
- Check input image preprocessing

## Part II: Transformer for Continuous Pose Estimation 