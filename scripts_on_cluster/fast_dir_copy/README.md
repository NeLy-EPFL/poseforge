# fast_dir_copy

Fast, multiprocessed copy of a directory tree on the cluster.

`parallel_copy.run` fans the per-file copies out across every CPU SLURM
allocates to the job (`xargs -P $SLURM_CPUS_PER_TASK`), which is dramatically
faster than a single-threaded `cp -r` / `rsync` on large datasets with many
files.

## Usage

```bash
sbatch parallel_copy.run <SRC_DIR> <DST_DIR>
```

Example:

```bash
sbatch parallel_copy.run /scratch/$USER/dataset /scratch/$USER/dataset_copy
```

## What it does

1. Recreates the source directory tree in the destination (avoids `mkdir` races).
2. Copies all regular files in parallel with `cp -a` (preserves attributes).
3. Copies symlinks verbatim.
4. Verifies source and destination entry counts match.

## Tuning

Edit the `#SBATCH` header in `parallel_copy.run` to change parallelism
(`--cpus-per-task`), wall time (`--time`), or memory (`--mem`).
