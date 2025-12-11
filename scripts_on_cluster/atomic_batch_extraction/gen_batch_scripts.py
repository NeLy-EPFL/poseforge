from pathlib import Path


# Batch script generation
template_path = Path("template.run")
batch_scripts_dir = Path("batch_scripts/")
batch_scripts_dir.mkdir(exist_ok=True, parents=True)

# Configs by task
synthetic_videos_basedir = Path("/home/sibwang/poseforge/bulk_data/style_transfer/production/translated_videos/")
trial_names_all = [x.name for x in synthetic_videos_basedir.glob("BO_Gal4_*")]

# Generate batch scripts
with open("template.run") as f:
    template_str = f.read()

for trial_name in trial_names_all:
    batch_script_str = template_str.replace("<<<TRIAL_NAME>>>", trial_name)
    with open(batch_scripts_dir / f"{trial_name}.run", "w") as f:
        f.write(batch_script_str)

print(f"Generated {len(trial_names_all)} batch scripts under {batch_scripts_dir}")
