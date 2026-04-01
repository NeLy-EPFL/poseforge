"""
Adapted from https://github.com/taesungp/contrastive-unpaired-translation/blob/master/train.py
"""

import json
from argparse import Namespace
from pathlib import Path
from cut.options.train_options import TrainOptions
from cut.data import create_dataset
from cut.util import util

import matplotlib.pyplot as plt

from poseforge.util import set_random_seed


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

    # Create a dataset given opt.dataset_mode and other options
    dataset = create_dataset(opt)
    dataset_size = len(dataset)

    # go through the dataset and imshow one batch just for debugging
    data = dataset.dataloader.__iter__().__next__()
    batch_size = data["A"].size(0)
    print(f"Batch size: {batch_size}")
    fig, axs = plt.subplots(2, batch_size, figsize=(5 * batch_size, 10))
    for i in range(batch_size):
        # `util.tensor2im` from CUT expects a batched tensor (N, C, H, W)
        image_A = util.tensor2im(data["A"][i : i + 1])
        image_B = util.tensor2im(data["B"][i : i + 1])
        axs[0, i].imshow(image_A)
        axs[0, i].set_title("Image A (input)")
        axs[0, i].axis("off")
        axs[1, i].imshow(image_B)
        axs[1, i].set_title("Image B (target)")
        axs[1, i].axis("off")
    plt.tight_layout()
    plt.show(block=True)

    print(image_A.shape, image_B.shape)