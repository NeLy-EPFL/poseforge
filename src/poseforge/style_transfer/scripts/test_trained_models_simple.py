from poseforge.style_transfer.scripts.test_trained_models import test_checkpoint
from pathlib import Path

def easy_test_checkpoint(checkpoint_path: str, simulation_data_dirs: str):
    campaign_name = "test"
    checkpoint_path = Path(checkpoint_path)
    run_name = "test"
    trial_name = checkpoint_path.parent.name
    epoch = int(checkpoint_path.stem.split("_")[0])
    output_basedir = checkpoint_path.parent.parent.parent.parent
    simulation_data_dirs = [Path(simulation_data_dirs).expanduser().resolve()]

    test_checkpoint(
        campaign_name=campaign_name,
        trial_name=trial_name,
        run_name=run_name,
        epoch=epoch,
        checkpoint_path=checkpoint_path,
        simulation_data_dirs=simulation_data_dirs,
        output_basedir=output_basedir,
    )

if __name__ == "__main__":
    import tyro
    tyro.cli(easy_test_checkpoint)