import yaml
from pathlib import Path
from flygym.compose.fly import FLYBODY_VISUALS_CONFIG_PATH


def represent_list(dumper, data):
    """Represent lists as flow-style YAML sequences."""
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)


def represent_str(dumper, data):
    """Represent strings to preserve quoting in YAML output."""
    if data in ['true', 'false'] or data.isdigit() or '.' in data:
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style="'")
    return dumper.represent_scalar('tag:yaml.org,2002:str', data)


yaml.add_representer(list, represent_list)
yaml.add_representer(tuple, represent_list)
yaml.add_representer(str, represent_str)


def generate_grayscale_config_flybody(base_config_path):
    """
    Generate a grayscale visualization configuration from a FlybodyFly base YAML file.
    
    This function converts color-based visual configurations to grayscale by applying
    the luminosity formula: gray = 0.299*R + 0.587*G + 0.114*B

    Args:
        base_config_path (Path): Path to the base FlybodyFly visual configuration file.

    Returns:
        dict: The grayscale visualization configuration.
    """
    with open(base_config_path, "r") as f:
        base_config = yaml.safe_load(f)

    grayscale_config = {}
    
    for vis_set_name, vis_set_data in base_config.items():
        # Create new visual set with grayscale prefix
        grayscale_vis_set_name = f"grayscale_{vis_set_name}"
        grayscale_config[grayscale_vis_set_name] = {}
        
        # Copy apply_to pattern as-is
        if "apply_to" in vis_set_data:
            grayscale_config[grayscale_vis_set_name]["apply_to"] = vis_set_data["apply_to"]
        
        # Convert material colors to grayscale
        if "material" in vis_set_data:
            grayscale_config[grayscale_vis_set_name]["material"] = {}
            
            for material_key, material_value in vis_set_data["material"].items():
                if material_key == "rgba" and material_value is not None:
                    # Convert RGBA strings to floats, compute grayscale, convert back to strings
                    rgba = [float(val) for val in material_value]
                    r, g, b, a = rgba[0], rgba[1], rgba[2], rgba[3]
                    gray = 0.299 * r + 0.587 * g + 0.114 * b
                    grayscale_config[grayscale_vis_set_name]["material"]["rgba"] = [
                        str(gray), str(gray), str(gray), str(a)
                    ]
                else:
                    # Preserve other material properties as-is
                    grayscale_config[grayscale_vis_set_name]["material"][material_key] = material_value
        
        # Convert texture colors to grayscale if present
        if "texture" in vis_set_data:
            grayscale_config[grayscale_vis_set_name]["texture"] = {}
            
            for texture_key, texture_value in vis_set_data["texture"].items():
                if texture_key in ["rgb1", "rgb2", "markrgb"] and texture_value is not None:
                    # Convert RGB strings to floats, compute grayscale, convert back to strings
                    rgb = [float(val) for val in texture_value]
                    r, g, b = rgb[0], rgb[1], rgb[2]
                    gray = 0.299 * r + 0.587 * g + 0.114 * b
                    grayscale_config[grayscale_vis_set_name]["texture"][texture_key] = [
                        str(gray), str(gray), str(gray)
                    ]
                else:
                    # Preserve other texture properties as-is
                    grayscale_config[grayscale_vis_set_name]["texture"][texture_key] = texture_value

    return grayscale_config


def main():
    """
    Generate and save grayscale visualization YAML file for FlybodyFly.
    """
    output_dir = Path(__file__).parent
    base_yaml_path = output_dir / "flybody_base.yaml"
    
    if not base_yaml_path.exists():
        # If flybody_base.yaml doesn't exist, use the default FlybodyFly config
        base_yaml_path = FLYBODY_VISUALS_CONFIG_PATH
        print(f"Using default FlybodyFly config: {base_yaml_path}")
    
    grayscale_config = generate_grayscale_config_flybody(base_yaml_path)
    
    output_path = output_dir / "flybody_grayscale.yaml"
    with open(output_path, "w") as f:
        yaml.dump(grayscale_config, f, default_flow_style=False, sort_keys=False, indent=2)
    print(f"Generated {output_path}")


if __name__ == "__main__":
    main()
