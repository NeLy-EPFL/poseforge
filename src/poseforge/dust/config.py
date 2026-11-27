"""Configuration objects for dust primitive extraction and compositing.

These mirror the ``SerializableDataClass`` style used elsewhere in poseforge so
they can be serialized alongside runs and reproduced exactly.
"""

from dataclasses import dataclass, field

from poseforge.util import SerializableDataClass


@dataclass(frozen=True)
class DustExtractionConfig(SerializableDataClass):
    """Parameters controlling how dust primitives are isolated from real frames.

    The imaging is dark-field: the background is near-black and everything bright
    is either the subject (fly) or junk. We remove the fly as the single largest
    bright connected component, isolate the remaining bright-above-background
    pixels, and cut each connected dust region out as one primitive.
    """

    # Low threshold offset above the per-frame background median used to find the
    # bright foreground (fly + junk). Kept low so dim wings join the fly body as
    # one connected component instead of leaking into the dust.
    foreground_threshold_offset: int = 18
    # Clamp range for the adaptive foreground threshold, in 8-bit intensity units.
    foreground_threshold_min: int = 25
    foreground_threshold_max: int = 45
    # Kernel size (pixels) to close small gaps so thin legs stay attached to the
    # body, without bridging the larger gaps to nearby detached junk.
    fly_close_kernel_size: int = 9
    # Kernel size (pixels) to dilate the fly mask before removing it, so the
    # diffuse out-of-focus halo around the animal is stripped along with it.
    fly_halo_dilate_kernel_size: int = 41
    # Intensity offset above the background median subtracted from the isolated
    # dust so only brightness genuinely above background remains.
    dust_floor_offset: int = 8
    dust_floor_min: int = 18
    dust_floor_max: int = 30
    # A connected dust region must have at least this many pixels to be kept
    # (rejects single-pixel sensor noise).
    min_primitive_area: int = 6
    # ... and at most this fraction of the frame area (rejects leftover fly
    # fragments and other oversized blobs that are not genuine dust).
    max_primitive_area_fraction: float = 0.03
    # Reject a primitive whose bounding box side exceeds this fraction of the
    # frame side (a long thin fly-leg remnant would be caught here).
    max_primitive_extent_fraction: float = 0.25
    # Reject a primitive whose bounding box lies within this fraction of the
    # frame border. Edge debris is often a wall fragment or crop-boundary
    # artifact, and since primitives are placed anywhere at compositing time we
    # only want cleanly-isolated ones. Set to 0 to keep edge primitives.
    edge_exclude_margin_fraction: float = 0.03
    # Whether to drop near-duplicate primitives (e.g. one static lens speck
    # recurring across many frames of a single recording).
    deduplicate: bool = True
    # Two primitives are considered duplicates when their resized intensity
    # signatures differ (mean absolute difference, 0-255) by less than this.
    deduplicate_tolerance: float = 6.0


@dataclass(frozen=True)
class LibraryBalanceConfig(SerializableDataClass):
    """Balance the primitive library across size classes.

    Extracting from many recordings yields an overwhelming number of tiny
    speckles. To keep the library compact and diverse, primitives are binned by
    pixel area and at most ``max_per_bin`` are kept per bin via reservoir
    sampling (a uniform random sample across all frames, in constant memory).
    """

    enabled: bool = True
    # Upper area edges (in pixels) of the size bins. A primitive with area A
    # falls in the first bin whose edge it does not exceed; anything larger than
    # the last finite edge goes in a final "large" bin. Defaults roughly map to
    # speck / small / medium / clump / fiber-or-big.
    area_bin_edges: tuple[int, ...] = (20, 60, 200, 800)
    # Maximum primitives kept per size bin.
    max_per_bin: int = 300
    # Seed for the reservoir sampling, so balanced extraction is reproducible.
    seed: int = 0


@dataclass(frozen=True)
class DustCompositingConfig(SerializableDataClass):
    """Parameters controlling how dust primitives are composited onto a frame.

    Dark-field junk between the lens and the subject scatters light, so it adds
    brightness wherever it falls -- background or fly alike. Compositing is
    therefore additive over the whole frame, which naturally produces dust that
    covers the fly.
    """

    # Inclusive range for the number of primitives overlaid on a frame. The count
    # is drawn uniformly per frame, giving the random dust level.
    min_layers: int = 0
    max_layers: int = 8
    # Per-primitive multiplicative intensity jitter (apparent focus / opacity).
    min_intensity_scale: float = 0.6
    max_intensity_scale: float = 1.0
    # Per-primitive size jitter.
    min_size_scale: float = 0.7
    max_size_scale: float = 1.6
    # Per-primitive defocus blur sigma (pixels), so pasted dust is never sharper
    # than the scene.
    min_blur_sigma: float = 0.6
    max_blur_sigma: float = 1.8
    # Whether to randomly flip primitives horizontally / vertically.
    random_flip: bool = True
    # Primitives are sampled with probability proportional to their integrated
    # brightness raised to this power. Used only when no measured size
    # distribution is supplied (see use_measured_distribution).
    sampling_power: float = 1.0
    # Probability of adding any dust to a frame. None (the default) uses the
    # measured natural dust-presence rate (fraction of real frames with dust);
    # set a value to override it manually.
    dust_probability: float | None = None
    # When True and dust statistics are supplied, the number of primitives per
    # frame and their sizes are drawn from the measured distributions (the default
    # behaviour). When False, the manual min_layers/max_layers and sampling_power
    # are used instead -- the manual override.
    use_measured_distribution: bool = True
    # Fraction of overlaid primitives forced to land near the subject centre, to
    # guarantee some dust covers the fly. The rest are placed uniformly.
    over_subject_fraction: float = 0.25


@dataclass(frozen=True)
class WallExtractionConfig(SerializableDataClass):
    """Parameters controlling how arena-wall frames are isolated.

    A *wall* is the bright chamber edge that appears near a frame border when the
    fly is close to it. Rather than modelling it as a shape, we reuse the dust
    pipeline's fly removal on the centred frames:

    1. the fly is the single largest bright connected component at a very low
       threshold (so its dim legs are included), and it must cover the frame
       centre (the recordings are centred on the fly);
    2. the frame is kept only when the fly component's area is within the expected
       range -- a wall *fused* (touching) to the fly inflates it, and those frames
       are rejected, so stored walls never touch the fly;
    3. the fly is removed (dilated to strip its halo) and the *residual full
       frame* is kept, holding the non-touching wall (and any dust); residuals
       with a sizeable border-touching structure are stored as wall samples.
    """

    # Very low bright-foreground threshold (median + offset), so the fly INCLUDING
    # its dim leg tips forms one connected component.
    low_threshold_offset: int = 6
    low_threshold_min: int = 12
    low_threshold_max: int = 22
    # Close small gaps so thin legs stay attached to the body.
    fly_close_kernel_size: int = 9
    # Dilate the fly mask before removing it, so the diffuse out-of-focus halo is
    # stripped along with the body.
    fly_halo_dilate_kernel_size: int = 41
    # Background floor subtracted from the residual (median + offset), clipped at
    # zero, so only brightness genuinely above background remains.
    floor_offset: int = 8
    floor_min: int = 18
    floor_max: int = 30
    # Keep a frame only when the fly component's area lies within this fraction
    # range of the whole frame. Below it, the fly was not cleanly detected; above
    # it, a wall is fused to the fly (touching), which we never want.
    min_fly_area_fraction: float = 0.12
    max_fly_area_fraction: float = 0.40
    # A residual counts as holding a wall when it has a connected component of at
    # least this fraction of the frame area that touches the border band (this
    # separates a real wall from stray dust specks).
    min_wall_area_fraction: float = 0.005
    # Width of the border band (fraction of the frame side) a wall must reach into.
    wall_border_margin_fraction: float = 0.03
    # Cap on the number of full-frame wall residuals kept, via reservoir sampling,
    # to bound the wall library size across a large dataset.
    max_walls: int = 500


@dataclass(frozen=True)
class WallDistanceConfig(SerializableDataClass):
    """Select wall-bearing frames by the fly's stage-derived distance to the wall.

    The recordings are centred on a tethered fly whose arena position is set by
    the stage. Using the arena's stage->physical calibration we recover the fly's
    physical position per frame and its distance to the nearest arena wall -- a
    far more reliable cue than image thresholding for whether a wall is in view
    and, crucially, how close it is to the fly. Only frames whose distance falls
    in ``[min_distance_mm, max_distance_mm]`` are used as wall sources: closer and
    the wall overlaps the fly, farther and no wall is in the crop. (Chosen from
    labelled examples: a wall is present and clear of the body in this band.)
    """

    min_distance_mm: float = 1.9
    max_distance_mm: float = 2.5
    # Index, within ``keypoints_xy_pre_alignment``, of the body-centre anchor the
    # frames are aligned on (post-alignment it is pinned to the frame centre).
    anchor_keypoint_index: int = 1
    # File names inside a trial's ``processed/`` folder.
    behavior_metadata_name: str = "behavior_frames_metadata.csv"
    alignment_transforms_name: str = "behavior_alignment_transforms.h5"
    aligned_video_name: str = "aligned_behavior_video.mkv"
    # Columns in the behavior metadata holding the stage position (mm).
    stage_x_column: str = "x_pos_mm_interp"
    stage_y_column: str = "y_pos_mm_interp"
    # Arena extent (mm) as [x_width, y_height]; by default read from the arena
    # folder's ``metadata.yaml`` (``arena_dim``). Set to override.
    arena_width_mm: float | None = None
    arena_height_mm: float | None = None


@dataclass(frozen=True)
class WallCompositingConfig(SerializableDataClass):
    """Parameters controlling how wall residuals are composited onto a frame.

    A wall's full-frame residual (fly removed) is *added* to the frame -- exactly
    like dust, dark-field brightness simply adds -- after mirroring on the two
    axes only (never rotated or translated), so it stays anchored to its border.
    No masking is applied: each residual already has a hole where its source fly
    was, so on the fly-centred frames a mirrored wall lands in the periphery.
    """

    # Per-wall intensity jitter and defocus blur (as for dust).
    min_intensity_scale: float = 0.7
    max_intensity_scale: float = 1.0
    min_blur_sigma: float = 0.6
    max_blur_sigma: float = 1.6
    # Probability of adding a wall to a frame. None (the default) uses the
    # measured natural wall-presence rate; set a value to override it manually.
    wall_probability: float | None = None
