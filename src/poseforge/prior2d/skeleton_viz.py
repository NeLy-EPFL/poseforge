"""Shared OpenCV-only (no matplotlib) skeleton drawing for pose visualization.

Used by `scripts/convert_slp.py` (annotated overview videos) and
`scripts/visualize_periods.py` (annotated period videos).
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


def build_skeleton(node_names):
    """Build (edges, point_colors) for the fly leg convention.

    Args:
        node_names: List of node names, in the same order as the pose array's
            node axis.

    Returns:
        edges: List of (idx_a, idx_b, color) tuples.
        point_colors: List of BGR colors, one per node, in node_names order.
    """
    name_to_idx = {name: i for i, name in enumerate(node_names)}

    edges = []
    for leg in LEG_PREFIXES:
        chain = [f"{leg}_{joint}" for joint in LEG_JOINTS]
        color = LEG_COLORS[leg]
        for a, b in itertools.pairwise(chain):
            if a in name_to_idx and b in name_to_idx:
                edges.append((name_to_idx[a], name_to_idx[b], color))

    leg_node_names = {f"{leg}_{joint}" for leg in LEG_PREFIXES for joint in LEG_JOINTS}
    point_colors = []
    for name in node_names:
        if name == "Th":
            point_colors.append(HUB_COLOR)
        elif name in leg_node_names:
            point_colors.append(LEG_COLORS[name.split("_")[0]])
        else:
            point_colors.append(OTHER_COLOR)

    return edges, point_colors


def draw_pose(frame, points, edges, point_colors):
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
            LINE_THICKNESS,
            cv2.LINE_AA,
        )
    for idx, pt in enumerate(points):
        if np.any(np.isnan(pt)):
            continue
        cv2.circle(
            frame,
            (round(pt[0]), round(pt[1])),
            POINT_RADIUS,
            point_colors[idx],
            -1,
            cv2.LINE_AA,
        )
    return frame
