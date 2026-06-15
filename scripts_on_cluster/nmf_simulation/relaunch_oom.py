#!/usr/bin/env python3
"""Detect OOM-killed SLURM jobs and regenerate their batch scripts from template.run."""

import argparse
import re
from pathlib import Path

OOM_PATTERN = re.compile(r"Detected \d+ oom_kill event")

SCRIPT_DIR = Path(__file__).parent
TEMPLATE_PATH = SCRIPT_DIR / "template.run"

# Parameters to extract from existing .run scripts
PARAM_PATTERNS = {
    "LOG_OUTPUT_FILE":    re.compile(r"#SBATCH --output (.+)"),
    "RECORDED_TRIAL_PATH": re.compile(r"--recorded-trial-path (.+)"),
    "TRIAL_OUTPUT_DIR":   re.compile(r"--trial-output-dir (.+)"),
    "SEGMENT_IDS":        re.compile(r"--segment-ids (.+)"),
    "USE_FLYBODY_FLAG":   re.compile(r"(--(?:use|no-use)-flybody)"),
}


def find_oom_logs(log_dirs):
    """Return (log_file, model) pairs where an OOM kill was detected."""
    results = []
    for log_dir in log_dirs:
        log_dir = Path(log_dir)
        if not log_dir.exists():
            print(f"Warning: {log_dir} does not exist, skipping.")
            continue
        model = log_dir.name.removeprefix("logs_")
        for log_file in sorted(log_dir.glob("*.out")):
            text = log_file.read_text(errors="replace")
            if OOM_PATTERN.search(text):
                results.append((log_file, model))
    return results


def extract_params(run_script: Path) -> dict:
    """Extract substitution values from an existing .run script."""
    text = run_script.read_text()
    params = {}
    for key, pattern in PARAM_PATTERNS.items():
        m = pattern.search(text)
        if m is None:
            raise ValueError(f"Could not find {key} in {run_script}")
        params[key] = m.group(1).strip()
    return params


def make_run_script(job_name: str, params: dict, template_str: str, output_dir: Path):
    script_str = (
        template_str
        .replace("<<<JOB_NAME>>>", job_name)
        .replace("<<<LOG_OUTPUT_FILE>>>", params["LOG_OUTPUT_FILE"])
        .replace("<<<RECORDED_TRIAL_PATH>>>", params["RECORDED_TRIAL_PATH"])
        .replace("<<<TRIAL_OUTPUT_DIR>>>", params["TRIAL_OUTPUT_DIR"])
        .replace("<<<SEGMENT_IDS>>>", params["SEGMENT_IDS"])
        .replace("<<<USE_FLYBODY_FLAG>>>", params["USE_FLYBODY_FLAG"])
    )
    dest = output_dir / f"{job_name}.run"
    dest.write_text(script_str)
    return dest


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate batch scripts for OOM-killed jobs from the current template.run. "
            "Edit template.run (e.g. bump --mem) before running this script."
        )
    )
    parser.add_argument(
        "output_folder",
        help=(
            "Destination folder for regenerated scripts "
            "(relative to this script's directory, e.g. 'retry_oom_highmem')."
        ),
    )
    parser.add_argument(
        "--log-dirs",
        nargs="+",
        default=None,
        help=(
            "Log directories to scan (default: all logs_* dirs next to this script). "
            "Example: --log-dirs logs_nmf logs_flybody"
        ),
    )
    args = parser.parse_args()

    log_dirs = (
        [Path(d) for d in args.log_dirs]
        if args.log_dirs
        else sorted(SCRIPT_DIR.glob("logs_*"))
    )

    template_str = TEMPLATE_PATH.read_text()
    print(f"Using template: {TEMPLATE_PATH}")
    print(f"Scanning log dirs: {[str(d) for d in log_dirs]}")

    oom_jobs = find_oom_logs(log_dirs)
    if not oom_jobs:
        print("No OOM-killed jobs found.")
        return

    output_dir = SCRIPT_DIR / args.output_folder
    if output_dir.exists() and list(output_dir.glob("*.run")):
        raise RuntimeError(
            f"{output_dir} already contains .run files. "
            "Empty it manually to avoid mixing stale scripts with retries."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    generated, missing = 0, []
    for log_file, model in oom_jobs:
        job_name = log_file.stem
        src = SCRIPT_DIR / f"batch_scripts_{model}" / f"{job_name}.run"
        if not src.exists():
            missing.append(str(src))
            print(f"  WARNING: original script not found — {src}")
            continue
        params = extract_params(src)
        dest = make_run_script(job_name, params, template_str, output_dir)
        print(f"  Generated [{model}]: {dest.name}")
        generated += 1

    print(f"\n{generated} script(s) written to {output_dir}")
    if missing:
        print(f"{len(missing)} original script(s) not found (already listed above).")


if __name__ == "__main__":
    main()
