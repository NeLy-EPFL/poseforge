import dataclasses
import yaml
import logging
from collections import defaultdict
from pathlib import Path
from typing import Hashable, Callable, Any

import poseforge


@dataclasses.dataclass(frozen=True)
class SerializableDataClass:
    def save(self, path: Path | str):
        if not Path(path).suffix.lower() in [".yaml", ".yml"]:
            raise ValueError(
                f"Invalid file extension for {path}. Expected .yaml or .yml"
            )
        with open(path, "w") as f:
            yaml.safe_dump(dataclasses.asdict(self), f, indent=2, sort_keys=False)

    @classmethod
    def load(cls, path: Path | str):
        """Load configuration from YAML file"""
        if not Path(path).is_file():
            raise FileNotFoundError(f"File does not exist: {path}")
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)


class OutputBuffer:
    """Buffer to hold incoming data for multiple datasets when the data
    arrive out of sync. Useful when handling per-frame output data from
    multiple videos that are loaded in parallel and processed in
    arbitrarily mixed batches."""

    def __init__(
        self,
        buckets_and_expected_sizes: dict[Hashable, int | None],
        closing_func: Callable[[Hashable, list[Any]], None],
    ):
        """
        Args:
            buckets_and_expected_sizes: Dictionary mapping bucket
                identifiers (any hashable type) to their expected size
                (e.g. number of frames). Buckets will be closed when they
                reach this size. If expected size is None, the bucket
                will never be closed automatically; instead, it must be
                closed manually by calling `close_bucket(bucket_id)`.
            closing_func: Function to call when a bucket is full and needs
                to be closed. It should take two arguments: the bucket
                identifier and a list of data items in the bucket (now
                sorted by index).
        """
        self.closing_func = closing_func
        self.expected_buckets_and_sizes = {}
        self.buffers = defaultdict(dict)

        for bucket, expected_size in buckets_and_expected_sizes.items():
            self.add_bucket(bucket, expected_size)

    def add_bucket(self, bucket: Hashable, expected_size: int | None):
        if bucket in self.expected_buckets_and_sizes:
            raise ValueError(f"Bucket {bucket} already exists")
        self.expected_buckets_and_sizes[bucket] = expected_size

    def clear_bucket(self, bucket: Hashable):
        if bucket in self.buffers:
            del self.buffers[bucket]

    def remove_bucket(self, bucket: Hashable):
        if bucket in self.expected_buckets_and_sizes:
            del self.expected_buckets_and_sizes[bucket]
        if bucket in self.buffers:
            del self.buffers[bucket]

    def is_bucket_full(self, bucket: Hashable) -> bool:
        if self.expected_buckets_and_sizes[bucket] is None:
            return False

        buf = self.buffers[bucket]
        expected_size = self.expected_buckets_and_sizes[bucket]

        # Quick length check first
        if len(buf) != expected_size:
            return False

        # Check if all indices 0..n-1 are present
        buf_indices = set(buf.keys())
        expected_indices = set(range(expected_size))
        if buf_indices != expected_indices:
            raise ValueError(
                f"Buffer for bucket {bucket} has reached the pre-specified size, "
                f"but the indices in the buffer are not contiguous."
            )

        return True

    def close_bucket(self, bucket: Hashable, force: bool = False):
        if bucket not in self.buffers:
            raise ValueError(f"Bucket {bucket} not found in buffers")
        if not self.is_bucket_full(bucket):
            if self.expected_buckets_and_sizes[bucket] is None:
                pass  # Always okay to close if expected size is None
            elif force:
                logging.warning(
                    f"Bucket {bucket} is not full yet, but force is True. Closing it "
                    "anyway."
                )
            else:
                raise ValueError(f"Bucket {bucket} is not full yet")

        sorted_indices = sorted(self.buffers[bucket].keys())
        data_items = [self.buffers[bucket][i] for i in sorted_indices]
        self.closing_func(bucket, data_items)
        del self.buffers[bucket]
        del self.expected_buckets_and_sizes[bucket]

    def close_all(self, force: bool = False):
        # Make a copy of the bucket names! Don't modify the dict while iterating over it
        all_buffered_buckets = list(self.buffers.keys())
        for bucket in all_buffered_buckets:
            self.close_bucket(bucket, force=force)

    def add_data(self, bucket: Hashable, index: int, data: Any):
        if bucket not in self.expected_buckets_and_sizes:
            raise ValueError(f"Bucket {bucket} not found in expected buckets")
        if index in self.buffers[bucket]:
            raise ValueError(
                f"Index {index} already exists in buffer for bucket {bucket}"
            )
        self.buffers[bucket][index] = data

        if self.is_bucket_full(bucket):
            self.close_bucket(bucket)
        else:
            expected_size = self.expected_buckets_and_sizes[bucket]
            if expected_size is not None:
                assert len(self.buffers[bucket]) < expected_size

    @property
    def n_open_buckets(self) -> int:
        return len(self.buffers)

    @property
    def n_data_total(self) -> int:
        return sum(len(buf) for buf in self.buffers.values())


assert len(poseforge.__path__) == 1, "poseforge.__path__ contains multiple paths"
bulk_data_dir = Path(poseforge.__path__[0]).parent.parent / "bulk_data"
