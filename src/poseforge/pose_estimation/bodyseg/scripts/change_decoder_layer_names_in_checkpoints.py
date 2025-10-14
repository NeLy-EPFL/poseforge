"""
Temporary script to update module names in checkpoint files. `decoder2`,
`decoder3` etc. have been renamed to `dec_layer2`, `dec_layer3` etc.
"""

import torch
import tyro
from collections import OrderedDict
from pathlib import Path


def rename_variables(src_path: Path, tgt_path: Path):
    ckpt = torch.load(src_path, map_location="cpu")
    ckpt_renamed = OrderedDict()
    for k, v in ckpt.items():
        if k.startswith("decoder"):
            new_k = k.replace("decoder", "dec_layer")
        else:
            new_k = k
        ckpt_renamed[new_k] = v
    torch.save(ckpt_renamed, tgt_path)


def process_all_checkpoints(src_dir: str, tgt_dir: str, /):
    src_dir = Path(src_dir)
    tgt_dir = Path(tgt_dir)
    tgt_dir.mkdir(exist_ok=True, parents=True)
    for src_path in src_dir.iterdir():
        if src_path.suffix.lower() not in (".pth", ".pt", ".ckpt"):
            continue
        print(f"processing {src_path}")
        tgt_path = tgt_dir / src_path.name
        rename_variables(src_path, tgt_path)


if __name__ == "__main__":
    tyro.cli(process_all_checkpoints)