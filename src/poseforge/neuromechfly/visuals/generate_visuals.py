import yaml
from pathlib import Path
from flygym.compose.fly import DEFAULT_RIGGING_CONFIG_PATH
from poseforge.neuromechfly.constants import (
    color_by_kinematic_chain,
    color_by_link,
    color_palette,
)

def represent_list(dumper, data):
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

yaml.add_representer(list, represent_list)
yaml.add_representer(tuple, represent_list)

def get_all_body_segments():
    """Load all body segment names from the default fly rigging configuration."""
    with open(DEFAULT_RIGGING_CONFIG_PATH, "r") as f:
        rigging_config = yaml.safe_load(f)
    return list(rigging_config.keys())

def generate_visual_config(body_segments, color_map, palette, prefix, default_color_name="gray"):
    """
    Generate a visualization configuration dictionary.

    Args:
        body_segments (list): A list of all body segment names.
        color_map (dict): A dictionary mapping segment keywords to color names.
        palette (dict): A dictionary mapping color names to RGBA tuples.
        default_color_name (str): The color name for segments that don't match any key in color_map.

    Returns:
        dict: The visualization configuration.
    """
    visual_config = {}
    assigned_segments = set()

    # Assign colors based on the color_map
    for key, color_name in color_map.items():
        apply_to_list = []
        # Match body segments to the current key
        # This handles both leg prefixes (e.g., "LF") and link names (e.g., "Coxa")
        for segment in body_segments:
            if key in segment and segment not in assigned_segments:
                apply_to_list.append(f"{segment}*")
        
        if apply_to_list:
            # Use a sanitized key for the visual set name
            vis_set_name = key.lower().replace("*", "").replace("_", "")
            vis_set_name = f"{prefix}_{vis_set_name}"
            visual_config[vis_set_name] = {
                "apply_to": apply_to_list,
                "material": {
                    "specular": 0.0,
                    "shininess": 0.0,
                    "rgba": list(palette[color_name]),
                },
            }
            assigned_segments.update(apply_to_list)

    # Assign default color to remaining segments
    unassigned_segments = [seg for seg in body_segments if seg not in assigned_segments]
    if unassigned_segments:
        visual_config[f"{prefix}_default"] = {
            "apply_to": [f"{seg}*" for seg in unassigned_segments],
            "material": {
                "specular": 0.0,
                "shininess": 0.0,
                "rgba": list(palette[default_color_name]),
            },
        }

    return visual_config

def generate_grayscale_config(base_config_path):
    """
    Generate a grayscale visualization configuration from a base YAML file.

    Args:
        base_config_path (Path): Path to the base visual configuration file.

    Returns:
        dict: The grayscale visualization configuration.
    """
    with open(base_config_path, "r") as f:
        base_config = yaml.safe_load(f)

    grayscale_config = {}
    for vis_set_name, vis_set_data in base_config.items():
        vis_set_name_with_prefix = f"grayscale_{vis_set_name}"
        grayscale_config[vis_set_name_with_prefix] = vis_set_data.copy()
        if "material" in vis_set_data and "rgba" in vis_set_data["material"]:
            r, g, b, a = vis_set_data["material"]["rgba"]
            gray = 0.299 * r + 0.587 * g + 0.114 * b
            grayscale_config[vis_set_name_with_prefix]["material"]["rgba"] = [gray, gray, gray, a]
        if "texture" in vis_set_data and "rgb1" in vis_set_data["texture"]:
            r, g, b = vis_set_data["texture"]["rgb1"]
            gray = 0.299 * r + 0.587 * g + 0.114 * b
            grayscale_config[vis_set_name_with_prefix]["texture"]["rgb1"] = [gray, gray, gray]
        if "texture" in vis_set_data and "rgb2" in vis_set_data["texture"]:
            r, g, b = vis_set_data["texture"]["rgb2"]
            gray = 0.299 * r + 0.587 * g + 0.114 * b
            grayscale_config[vis_set_name_with_prefix]["texture"]["rgb2"] = [gray, gray, gray]
        if "texture" in vis_set_data and "markrgb" in vis_set_data["texture"]:
            r, g, b = vis_set_data["texture"]["markrgb"]
            gray = 0.299 * r + 0.587 * g + 0.114 * b
            grayscale_config[vis_set_name_with_prefix]["texture"]["markrgb"] = [gray, gray, gray]

    return grayscale_config

def main():
    """
    Generate and save visualization YAML files.
    """
    output_dir = Path(__file__).parent
    body_segments = get_all_body_segments()

    # Generate config for per-leg coloring
    # We remove Thorax from the map as it's not a leg kinematic chain
    leg_color_map = {k: v for k, v in color_by_kinematic_chain.items() if k != "Thorax"}
    per_leg_config = generate_visual_config(body_segments, leg_color_map, color_palette, "leg")
    with open(output_dir / "per_leg_color.yaml", "w") as f:
        yaml.dump(per_leg_config, f, default_flow_style=False, sort_keys=False, indent=2)
    print(f"Generated {output_dir / 'per_leg_color.yaml'}")

    # Generate config for per-link coloring
    # Add a wildcard to match all tarsus segments
    link_color_map = color_by_link.copy()
    if "Tarsus" in link_color_map:
        link_color_map["*tarsus*"] = link_color_map.pop("Tarsus")

    per_link_config = generate_visual_config(body_segments, link_color_map, color_palette, "link")
    with open(output_dir / "per_link_color.yaml", "w") as f:
        yaml.dump(per_link_config, f, default_flow_style=False, sort_keys=False, indent=2)
    print(f"Generated {output_dir / 'per_link_color.yaml'}")

    # Generate grayscale config from base.yaml
    base_yaml_path = output_dir / "base.yaml"
    grayscale_config = generate_grayscale_config(base_yaml_path)
    with open(output_dir / "grayscale.yaml", "w") as f:
        yaml.dump(grayscale_config, f, default_flow_style=False, sort_keys=False, indent=2)
    print(f"Generated {output_dir / 'grayscale.yaml'}")


if __name__ == "__main__":
    main()