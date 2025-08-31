"""
Adapted from https://github.com/taesungp/contrastive-unpaired-translation/blob/master/train.py
"""

import time
import torch
import numpy as np
import wandb
from cut.options.train_options import TrainOptions
from cut.data import create_dataset
from cut.models import create_model
from cut.util.visualizer import Visualizer
from cut.util import util

from biomechpose.util import set_random_seed


if __name__ == "__main__":
    # Set random seed for reproducibility
    set_random_seed(42)

    # Get training options
    opt = TrainOptions().parse()
    total_num_epochs = opt.n_epochs + opt.n_epochs_decay

    # Create a dataset given opt.dataset_mode and other options
    dataset = create_dataset(opt)
    dataset_size = len(dataset)

    # Create a model given opt.model and other options
    model = create_model(opt)
    print(f"The number of training images = {dataset_size}")

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

            # Display images on tensorboard and save images
            if total_iters % opt.display_freq == 0:
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

    # Close TensorBoard writer when training finishes
    if hasattr(visualizer, "close"):
        visualizer.close()
    if opt.wandb_project:
        wandb.finish()
