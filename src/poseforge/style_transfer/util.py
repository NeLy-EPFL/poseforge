import re
import logging
from typing import Any


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
