"""Shared OpenCV-only (no matplotlib) skeleton drawing for pose visualization.

Used by `scripts/convert_slp.py` (annotated overview videos) and
`scripts/make_videos.py` (annotated period videos).
"""

import itertools

import cv2
import numpy as np

# Legs are connected as: *_ThC -> *_CTr -> *_FTi -> *_TiTa -> *_Cl. ThC is not
# connected to the thorax (Th): unlike the other joints, the real skeleton
# only wires Th to the middle legs' ThC, not the front/hind ones, so drawing
# a Th-ThC edge for every leg here would misrepresent the real topology.
# Everything else (Th itself aside) is drawn as an unconnected dot.
LEG_PREFIXES = ["LF", "LM", "LH", "RF", "RM", "RH"]
LEG_JOINTS = ["ThC", "CTr", "FTi", "TiTa", "Cl"]

# BGR colors, one per leg chain.
LEG_COLORS = {
    "LF": (255, 80, 80),
    "LM": (80, 220, 80),
    "LH": (60, 60, 230),
    "RF": (230, 220, 60),
    "RM": (220, 80, 220),
    "RH": (60, 200, 230),
}
HUB_COLOR = (255, 255, 255)  # Th (thorax hub)
OTHER_COLOR = (180, 180, 180)  # N, A, LA, RA, LW, RW

POINT_RADIUS = 5
LINE_THICKNESS = 2


def build_skeleton(
    node_names, leg_colors=None, hub_color=HUB_COLOR, other_color=OTHER_COLOR
):
    """Build (edges, point_colors) for the fly leg convention.

    Args:
        node_names: List of node names, in the same order as the pose array's
            node axis.
        leg_colors: Dict from leg prefix (`LEG_PREFIXES`) to a BGR color.
            Defaults to `LEG_COLORS`; pass a different mapping (e.g.
            `poseforge.neuromechfly.constants.kchain_plotting_colors`,
            converted to BGR 0-255 tuples) to match another convention.
        hub_color: BGR color for the thorax ("Th") hub.
        other_color: BGR color for non-leg nodes.

    Returns:
        edges: List of (idx_a, idx_b, color) tuples.
        point_colors: List of BGR colors, one per node, in node_names order.
    """
    leg_colors = LEG_COLORS if leg_colors is None else leg_colors
    name_to_idx = {name: i for i, name in enumerate(node_names)}

    edges = []
    for leg in LEG_PREFIXES:
        chain = [f"{leg}_{joint}" for joint in LEG_JOINTS]
        color = leg_colors[leg]
        for a, b in itertools.pairwise(chain):
            if a in name_to_idx and b in name_to_idx:
                edges.append((name_to_idx[a], name_to_idx[b], color))

    leg_node_names = {f"{leg}_{joint}" for leg in LEG_PREFIXES for joint in LEG_JOINTS}
    point_colors = []
    for name in node_names:
        if name == "Th":
            point_colors.append(hub_color)
        elif name in leg_node_names:
            point_colors.append(leg_colors[name.split("_")[0]])
        else:
            point_colors.append(other_color)

    return edges, point_colors


def build_monochrome_skeleton(node_names, leg_color, other_color=OTHER_COLOR):
    """Build (edges, point_colors) with the whole leg skeleton in one color.

    Unlike `build_skeleton` (one color per leg), this uses a single color for
    every leg edge/point (and the thorax hub), e.g. to overlay raw
    predictions and IK forward-kinematics results in different colors on the
    same frame.

    Args:
        node_names: List of node names, in the same order as the pose array's
            node axis.
        leg_color: BGR color for all leg edges/points and the thorax hub.
        other_color: BGR color for non-leg nodes.

    Returns:
        edges: List of (idx_a, idx_b, color) tuples.
        point_colors: List of BGR colors, one per node, in node_names order.
    """
    name_to_idx = {name: i for i, name in enumerate(node_names)}

    edges = []
    for leg in LEG_PREFIXES:
        chain = [f"{leg}_{joint}" for joint in LEG_JOINTS]
        for a, b in itertools.pairwise(chain):
            if a in name_to_idx and b in name_to_idx:
                edges.append((name_to_idx[a], name_to_idx[b], leg_color))

    leg_node_names = {f"{leg}_{joint}" for leg in LEG_PREFIXES for joint in LEG_JOINTS}
    point_colors = [
        leg_color if (name == "Th" or name in leg_node_names) else other_color
        for name in node_names
    ]
    return edges, point_colors


def draw_pose(
    frame,
    points,
    edges,
    point_colors,
    line_thickness=LINE_THICKNESS,
    point_radius=POINT_RADIUS,
):
    """Draw skeleton edges and keypoint dots onto frame, in place."""
    for a, b, color in edges:
        pa, pb = points[a], points[b]
        if np.any(np.isnan(pa)) or np.any(np.isnan(pb)):
            continue
        cv2.line(
            frame,
            (round(pa[0]), round(pa[1])),
            (round(pb[0]), round(pb[1])),
            color,
            line_thickness,
            cv2.LINE_AA,
        )
    for idx, pt in enumerate(points):
        if np.any(np.isnan(pt)):
            continue
        cv2.circle(
            frame,
            (round(pt[0]), round(pt[1])),
            point_radius,
            point_colors[idx],
            -1,
            cv2.LINE_AA,
        )
    return frame
