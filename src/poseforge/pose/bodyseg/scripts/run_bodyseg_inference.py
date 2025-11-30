import torch
import h5py
from loguru import logger
from pathlib import Path
from torchvision.transforms import Resize
from torchsummary import summary
from tqdm import tqdm
from pvio.torch_tools import SimpleVideoCollectionLoader

import poseforge.pose.bodyseg.config as config
from poseforge.pose.bodyseg import BodySegmentationModel, BodySegmentationPipeline
from poseforge.util.sys import get_hardware_availability


def test_bodyseg_model(
    *,
    input_dir: Path,
    model_dir: Path,
    model_checkpoint_path: Path,
    save_prob: bool = False,
    output_basedir: Path | None = None,
    batch_size: int = 512,
    n_workers: int = 16,
    inference_image_size: tuple[int, int] = (256, 256),
):
    # System setup
    hardware_avail = get_hardware_availability(check_gpu=True, print_results=True)
    if len(hardware_avail["gpus"]) == 0:
        raise RuntimeError("No GPU available for testing")
    torch.backends.cudnn.benchmark = True

    # Create dataset and dataloader
    input_img_dir = input_dir / "model_prediction/not_flipped/"
    if len(list(input_img_dir.iterdir())) == 0:
        logger.warning(f"Trial {input_img_dir} is empty - skipping")
        return
    transform = Resize(inference_image_size)
    dataloader = SimpleVideoCollectionLoader(
        [input_img_dir],
        transform=transform,
        batch_size=batch_size,
        num_workers=n_workers,
    )
    n_frames = len(dataloader.dataset)
    logger.info(f"Found {n_frames} frames to process")
    logger.info(
        f"Using batch size {dataloader.batch_size} with {dataloader.num_workers} "
        f"workers. This will generate {len(dataloader)} batches."
    )

    # Create model and learning pipeline
    architecture_config_path = model_dir / "configs/model_architecture_config.yaml"
    model_weights = config.ModelWeightsConfig(model_weights=model_checkpoint_path)
    model = BodySegmentationModel.create_architecture_from_config(
        architecture_config_path
    ).cuda()
    model.load_weights_from_config(model_weights)
    summary(model, (3, *inference_image_size))
    pipeline = BodySegmentationPipeline(model, device="cuda", use_float16=True)

    # Set up output H5 files
    exp_trial_name = input_img_dir.parts[-3]
    out_dir = output_basedir / exp_trial_name
    out_dir.mkdir(parents=True, exist_ok=True)
    f_pred = h5py.File(out_dir / f"bodyseg_pred.h5", "w")
    f_prob = h5py.File(out_dir / f"bodyseg_prob.h5", "w") if save_prob else None
    ds_segmap = f_pred.create_dataset(
        "pred_segmap",
        shape=(n_frames, inference_image_size[0], inference_image_size[1]),
        dtype="uint8",
        compression="gzip",
        shuffle=True,
    )
    ds_segmap.attrs["class_labels"] = pipeline.class_labels
    ds_conf = f_pred.create_dataset(
        "pred_confidence",
        shape=(n_frames, inference_image_size[0], inference_image_size[1]),
        dtype="uint8",
        compression="gzip",
        shuffle=True,
    )
    ds_conf.attrs["scale"] = 100  # transform from 0-1 to 0-100 for uint8 storage
    ds_conf.attrs["method"] = model.confidence_method
    if save_prob:
        ds_probs = f_prob.create_dataset(
            "pred_probabilities",
            shape=(
                n_frames,
                model.n_classes,
                inference_image_size[0],
                inference_image_size[1],
            ),
            dtype="uint8",
            compression="gzip",
            shuffle=True,
        )
        ds_probs.attrs["scale"] = 100  # transform from 0-1 to 0-100 for uint8 storage

    # Inference loop
    log_interval = max(len(dataloader) // 10, 1)
    for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        # Forward pass (no need to move data to and from the GPU; pipeline will do that)
        pred_dict = pipeline.inference(batch["frames"])
        logits = pred_dict["logits"]

        # Save outputs
        frame_ids = batch["frame_indices"]
        pred_seg = torch.argmax(logits, dim=1).to(torch.uint8).detach().cpu()
        conf = (pred_dict["confidence"] * 100).to(torch.uint8).detach().cpu()
        ds_segmap[frame_ids, :, :] = pred_seg.numpy()
        ds_conf[frame_ids, :, :] = conf.numpy()
        if save_prob:
            prob = (torch.softmax(logits, dim=1) * 100).to(torch.uint8).detach().cpu()
            ds_probs[frame_ids, :, :, :] = prob.numpy()

        if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == len(dataloader):
            logger.info(f"Processed batch {batch_idx + 1} / {len(dataloader)}")

    logger.info("Inference complete")


if __name__ == "__main__":
    from poseforge.util.sys import set_loguru_level

    set_loguru_level("INFO")

    input_basedir = Path("bulk_data/behavior_images/spotlight_aligned_and_cropped/")
    model_dir = Path("bulk_data/pose_estimation/bodyseg/trial_20251127a")
    batch_size = 192
    n_workers = 16
    inference_image_size = (256, 256)
    output_buffer_log_interval = 10
    epoch = 8  # chosen by validation performance and visual inspection
    step = 18335  # last step of each epoch

    model_checkpoint_path = model_dir / f"checkpoints/epoch{epoch}_step{step}.model.pth"
    output_basedir = model_dir / f"production/epoch{epoch}_step{step}/"
    output_basedir.mkdir(parents=True, exist_ok=True)
    for input_dir in sorted(input_basedir.iterdir()):
        print(f"Processing {input_dir}")
        test_bodyseg_model(
            input_dir=input_dir,
            model_dir=model_dir,
            model_checkpoint_path=model_checkpoint_path,
            output_basedir=output_basedir,
            batch_size=batch_size,
            n_workers=n_workers,
            inference_image_size=inference_image_size,
            save_prob=True,
        )
