from pathlib import Path


if __name__ == "__main__":
    # Input dir
    all_preprocessed_images_dir = Path(
        "bulk_data/style_transfer/kinematic_recording/spotlight202506/spotlight_recordings_all_preprocessed"
    )
    # Output dir
    unified_images_dir = Path(
        "bulk_data/style_transfer/kinematic_recording/spotlight202506/spotlight_recordings_unified"
    )

    linked_count = 0
    exist_count = 0

    trial_dirs = sorted(list(all_preprocessed_images_dir.iterdir()))
    test_trials_dirs = trial_dirs[-1:]
    for trial_dir in trial_dirs:
        # Process each image in the trial directory
        for image_path in trial_dir.glob("*.jpg"):
            trial_name = trial_dir.name
            # Create a symbolic link to the image in the output directory
            tag = "test" if trial_dir in test_trials_dirs else "train_and_validate"
            new_image_path = (
                unified_images_dir / tag / (trial_name + "_" + image_path.name)
            )
            new_image_path.parent.mkdir(parents=True, exist_ok=True)
            if not new_image_path.exists():
                new_image_path.symlink_to(image_path.resolve())
                # print(f"Linked {image_path} to {new_image_path}")
                linked_count += 1
            else:
                # print(f"Link already exists: {new_image_path}")
                exist_count += 1

    print(f"Linked {linked_count} images to {unified_images_dir}")
    print(f"Skipped {exist_count} images that already exist in {unified_images_dir}")
