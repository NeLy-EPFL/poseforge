"""
Adapted from https://github.com/taesungp/contrastive-unpaired-translation/blob/master/train.py
"""

import copy
import random
import time
import torch
import numpy as np
import wandb
import json
from argparse import Namespace
from pathlib import Path
from torchvision.utils import make_grid
from cut.options.train_options import TrainOptions
from cut.data import create_dataset
from cut.models import create_model
from cut.util.visualizer import Visualizer
from cut.util import util

from poseforge.util import set_random_seed


VAL_EVERY_N_EPOCHS = 10
VAL_MAX_SAMPLES = 16
VAL_SEED = 12345


def save_options(
    train_options_obj: TrainOptions, parsed_opt: Namespace, output_path: Path | str
):
    output_path = Path(output_path)
    options_dict = {"values": {}, "defaults": {}}
    for key, val in sorted(vars(parsed_opt).items()):
        default = train_options_obj.parser.get_default(key)
        options_dict["values"][key] = val
        options_dict["defaults"][key] = default

    output_path.parent.mkdir(exist_ok=True, parents=True)
    with open(output_path, "w") as f:
        json.dump(options_dict, f, indent=2, sort_keys=True)


def build_image_logging_schedule(
    total_epochs: int, max_images: int = 50, dense_until: int = 20
) -> set[int]:
    """Build a set of epoch indices to log images for.

    Keeps early training dense and then increases stride so total image logs stay bounded.
    """
    if total_epochs <= 0:
        return set()

    max_images = max(1, max_images)
    if total_epochs <= max_images:
        return set(range(1, total_epochs + 1))

    dense_until = min(dense_until, total_epochs, max_images)
    scheduled = set(range(1, dense_until + 1))

    remaining = max_images - len(scheduled)
    if remaining > 0 and dense_until < total_epochs:
        tail = np.geomspace(dense_until + 1, total_epochs, num=remaining)
        scheduled.update(int(round(x)) for x in tail)

    scheduled.add(total_epochs)
    return scheduled


if __name__ == "__main__":
    # Set random seed for reproducibility
    set_random_seed(42)

    # Get training options and save a copy
    train_options_obj = TrainOptions()
    opt: Namespace = train_options_obj.parse()
    opt_file_path = Path(opt.checkpoints_dir) / "train_options.json"
    save_options(train_options_obj, opt, opt_file_path)

    # Compute total number of epochs to train
    total_num_epochs = opt.n_epochs + opt.n_epochs_decay
    image_log_epochs = build_image_logging_schedule(total_num_epochs, max_images=50)

    # Create a dataset given opt.dataset_mode and other options
    dataset = create_dataset(opt)
    dataset_size = len(dataset)

    # If the mask-aware edge loss is enabled, refuse to start without masks --
    # silently running with a missing mask directory would let the loss be a
    # no-op and waste a 24h training slot.
    if getattr(opt, "lambda_edge", 0.0) > 0.0:
        underlying_dataset = dataset.dataset
        assert getattr(underlying_dataset, "has_mask_A", False), (
            f"--lambda_edge={opt.lambda_edge} requires per-frame masks for "
            f"domain A, but no '{opt.phase}A_mask/' directory was found under "
            f"{opt.dataroot}. Re-extract the dataset with the updated "
            "extract_dataset.py (which now also writes silhouette masks)."
        )

    # Create a model given opt.model and other options
    model = create_model(opt)
    print(f"The number of training images = {dataset_size}")

    # Validation loader: reads testA/testB with the same preprocessing as
    # training, in deterministic alphabetical order. max_dataset_size is left
    # unbounded so that the alphabetical sort runs on the full file list; we
    # then take the first VAL_MAX_SAMPLES as a single batch.
    val_opt = copy.copy(opt)
    val_opt.phase = "test"
    val_opt.serial_batches = True
    val_opt.no_flip = True
    val_opt.batch_size = VAL_MAX_SAMPLES
    val_opt.max_dataset_size = float("inf")
    val_opt.num_threads = 0
    val_opt.isTrain = False
    val_dataset = create_dataset(val_opt)

    # Create a visualizer that display/save images and plots
    visualizer = Visualizer(opt)
    opt.visualizer = visualizer
    total_iters = 0  # the total number of training iterations

    optimize_time = 0.1

    # Initialize wandb if project name is provided
    if opt.wandb_project:
        wandb.init(project=opt.wandb_project, name=opt.name, config=vars(opt))

    times = []
    # Outer loop for different epochs; we save the model by
    # <epoch_count>, <epoch_count>+<save_latest_freq>
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        # Start timers
        epoch_start_time = time.time()  # for entire epoch
        iter_data_time = time.time()  # for data loading per iteration
        # Training iteration counter in current epoch, reset to 0 every epoch
        epoch_iter = 0
        # Reset the visualizer: make sure it saves the results at least once every epoch
        visualizer.reset()

        dataset.set_epoch(epoch)
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            batch_size = data["A"].size(0)
            total_iters += batch_size
            epoch_iter += batch_size
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            optimize_start_time = time.time()

            if epoch == opt.epoch_count and i == 0:
                model.data_dependent_initialize(data)
                # Regular setup: load and print networks; create schedulers
                model.setup(opt)
                model.parallelize()

            # Unpack data from dataset and apply preprocessing
            model.set_input(data)

            # Calculate loss functions, get gradients, update network weights
            model.optimize_parameters()

            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            optimize_time = (
                time.time() - optimize_start_time
            ) / batch_size * 0.005 + 0.995 * optimize_time

            # Display images only on scheduled epochs to avoid keeping only the
            # most recent dense image logs in TensorBoard.
            should_log_epoch_image = epoch in image_log_epochs and i == 0
            if should_log_epoch_image:
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visuals = model.get_current_visuals()
                visualizer.display_current_results(visuals, epoch, save_result)
                # Log images to wandb (concatenated if possible)
                concat_keys = ["real_A", "fake_B", "real_B", "idt_B"]
                images_to_concat = []
                for k in concat_keys:
                    if k in visuals:
                        image_numpy = util.tensor2im(visuals[k])
                        images_to_concat.append(image_numpy)
                if opt.wandb_project and len(images_to_concat) > 0:
                    concat_image = (
                        np.concatenate(images_to_concat, axis=1)
                        if len(images_to_concat) > 1
                        else images_to_concat[0]
                    )
                    img = wandb.Image(
                        concat_image, caption=f"Epoch {epoch}, Iter {total_iters}"
                    )
                    wandb.log({"results": [img]}, step=total_iters)

            # Print training losses and save logging information to the disk
            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                visualizer.print_current_losses(
                    epoch, epoch_iter, losses, optimize_time, t_data
                )
                if opt.display_id is None or opt.display_id > 0:
                    visualizer.plot_current_losses(
                        epoch, float(epoch_iter) / dataset_size, losses
                    )
                if opt.wandb_project:
                    wandb.log(
                        {**losses, "epoch": epoch, "iter": epoch_iter}, step=total_iters
                    )

            # Cache our latest model every <save_latest_freq> iterations
            if total_iters % opt.save_latest_freq == 0:
                print(
                    f"saving the latest model "
                    f"(epoch {epoch}, total_iters {total_iters})"
                )
                # Occasionally show the experiment name on console
                print(opt.name)
                save_suffix = f"iter_{total_iters}" if opt.save_by_iter else "latest"
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        # Cache our model every <save_epoch_freq> epochs
        if epoch % opt.save_epoch_freq == 0:
            print(f"saving the model at the end of epoch {epoch}, iters {total_iters}")
            model.save_networks("latest")
            model.save_networks(epoch)

        time_taken = int(time.time() - epoch_start_time)
        print(
            f"End of epoch {epoch} / {total_num_epochs} \t Time taken: {time_taken} sec"
        )
        # Update learning rates at the end of every epoch
        model.update_learning_rate()

        # Validation pass on testA: deterministic image order, same
        # preprocessing as training, identical crops across epochs and runs.
        # The global RNG state is snapshotted and restored so training's
        # random sequence is unaffected.
        if epoch % VAL_EVERY_N_EPOCHS == 0:
            py_rng_state = random.getstate()
            np_rng_state = np.random.get_state()
            torch_rng_state = torch.get_rng_state()
            random.seed(VAL_SEED)
            np.random.seed(VAL_SEED)
            torch.manual_seed(VAL_SEED)

            try:
                model.netG.eval()
                with torch.no_grad():
                    val_batch = next(iter(val_dataset))
                    real_A = val_batch["A"].to(model.device)
                    fake_B = model.netG(real_A).cpu()
                model.netG.train()
                pairs = []
                for source, fake in zip(real_A.cpu(), fake_B):
                    pairs.append(source)
                    pairs.append(fake)
                val_grid = make_grid(
                    pairs, nrow=2, padding=4,
                    normalize=True, value_range=(-1, 1),
                )
                if visualizer.writer is not None:
                    visualizer.writer.add_image("validation/grid", val_grid, epoch)
                if opt.wandb_project:
                    wandb.log(
                        {"validation/grid": wandb.Image(val_grid)},
                        step=total_iters,
                    )
            finally:
                random.setstate(py_rng_state)
                np.random.set_state(np_rng_state)
                torch.set_rng_state(torch_rng_state)

    # Close TensorBoard writer when training finishes
    if hasattr(visualizer, "close"):
        visualizer.close()
    if opt.wandb_project:
        wandb.finish()
