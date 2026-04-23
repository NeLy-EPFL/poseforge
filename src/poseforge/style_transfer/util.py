import re
import logging
from typing import Any
from pathlib import Path
import json


def parse_hyperparameters_from_trial_name(trial_name: str) -> dict[str, Any] | None:
    """Given the name of a training trial, parse its hyperparameters and
    return them as a dictionary."""
    trial_name_regex = r"ngf(?P<ngf>\d+)_netG(?P<netG>[a-zA-Z0-9]+)_batsize(?P<batsize>\d+)_lambGAN(?P<lambGAN>[\d.]+)"
    match = re.match(trial_name_regex, trial_name)
    if match:
        return {
            "ngf": int(match.group("ngf")),
            "netG": match.group("netG"),
            "batsize": int(match.group("batsize")),
            "lambGAN": float(match.group("lambGAN")),
        }
    else:
        logging.warning(f"Could not parse parameters from trial name: {trial_name}")
        return None

def parse_hyperparameters_from_checkpoint_path(checkpoint_path: Path) -> dict:
    """Given a checkpoint path, parse the hyperparameters from the trial name
    and return them as a dictionary."""
    train_opt_path = checkpoint_path.parent.parent / "train_options.json"
    if train_opt_path.is_file():
        train_option = {}
        train_option["preprocess_opt"] = {}
        with open(train_opt_path, "r") as f:
            full_train_option = json.load(f)
            train_option["netG"] = full_train_option["values"]["netG"]
            train_option["ngf"] = full_train_option["values"]["ngf"]
            train_option["image_side_length"] = full_train_option["values"]["crop_size"]

            train_option["preprocess_opt"]["preprocess"] = full_train_option["values"]["preprocess"]
            train_option["preprocess_opt"]["crop_size"] = full_train_option["values"]["crop_size"]
            load_size = full_train_option["values"]["load_size"]
            if full_train_option["values"]["n_epochs_decay"] > 0 and "finetune_load_size" in full_train_option["values"]:
                load_size = full_train_option["values"]["finetune_load_size"]
            train_option["preprocess_opt"]["load_size"] = load_size

        return train_option
            
    else:
        raise FileNotFoundError(
            f"Could not find train_options.json at expected path: {train_opt_path}"
        )
