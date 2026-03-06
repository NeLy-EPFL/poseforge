import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import time
from sklearn.decomposition import PCA
from scipy.interpolate import interp1d
from pathlib import Path


class LatentSpaceTrajectoryVisualizer:
    def __init__(
        self,
        trajectories: np.ndarray,
        trail_length: int = 50,
        fps: int = 30,
        headless: bool = False,
    ):
        """
        Initialize animator for multiple trajectories.

        Args:
            trajectories (np.ndarray): np.array of shape (n_traj, n_timesteps, 3)
            trail_length (int): Number of past positions to show in trail
            fps (int): Frames per second for animation (default: 30)
            headless (bool): Whether to run in offscreen mode (no display)
        """
        self.trajectories = trajectories
        self.n_traj, self.n_timesteps, _ = trajectories.shape
        self.trail_length = trail_length
        self.fps = fps
        self._delay = 1.0 / fps  # Calculate delay from FPS
        self.headless = headless

        # Calculate data bounds for automatic scaling
        self.data_bounds = self._calculate_data_bounds()
        self.data_center = self._calculate_data_center()
        self.data_scale = self._calculate_data_scale()

        # Allow empty meshes
        pv.global_theme.allow_empty_mesh = True

        # Create plotter with offscreen mode if needed
        self.plotter = pv.Plotter(off_screen=headless)
        self.plotter.set_background("black")

        # Set camera position based on data scale
        camera_distance = self.data_scale * 2.5
        self.plotter.camera.position = (
            self.data_center[0] + camera_distance,
            self.data_center[1] + camera_distance,
            self.data_center[2] + camera_distance * 0.6,
        )
        self.plotter.camera.focal_point = self.data_center

        # Add XYZ grid and axes with data-appropriate bounds
        # self.plotter.show_axes()
        self.plotter.show_grid(
            color="gray",
            show_xlabels=False,
            show_ylabels=False,
            show_zlabels=False,
            xtitle="PC 1",
            ytitle="PC 2",
            ztitle="PC 3",
            bounds=self.data_bounds,
        )

        # Create different colors for each trajectory
        self.colors = self._generate_colors(self.n_traj)

        # Pre-create trail meshes and point meshes for each trajectory
        self.trail_meshes = []
        self.point_meshes = []
        self.trail_actors = []
        self.point_actors = []

        for i in range(self.n_traj):
            # Create trail mesh
            trail_mesh = pv.PolyData()
            self.trail_meshes.append(trail_mesh)

            # Create point mesh
            point_mesh = pv.PolyData()
            self.point_meshes.append(point_mesh)

            # Add trail actor
            trail_actor = self.plotter.add_mesh(
                trail_mesh, color=self.colors[i], line_width=3
            )
            self.trail_actors.append(trail_actor)

            # Add point actor
            point_actor = self.plotter.add_mesh(
                point_mesh,
                color=self.colors[i],
                point_size=8,
                render_points_as_spheres=True,
                style="points",
            )
            self.point_actors.append(point_actor)

    def _generate_colors(self, n_colors):
        """Generate distinct colors for each trajectory"""
        cmap = plt.get_cmap("tab10")  # Use tab10 colormap for distinct colors
        colors = []
        for i in range(n_colors):
            rgb = cmap(i / max(1, n_colors - 1))[:3]  # Get RGB values
            colors.append(rgb)
        return colors

    def _calculate_data_bounds(self):
        """Calculate the bounding box of all trajectory data"""
        # Flatten all trajectories to find global min/max
        all_points = self.trajectories.reshape(-1, 3)

        min_bounds = np.min(all_points, axis=0)
        max_bounds = np.max(all_points, axis=0)

        # Add some padding (10% on each side)
        padding = (max_bounds - min_bounds) * 0.1
        min_bounds -= padding
        max_bounds += padding

        # Return as [xmin, xmax, ymin, ymax, zmin, zmax]
        return [
            min_bounds[0],
            max_bounds[0],
            min_bounds[1],
            max_bounds[1],
            min_bounds[2],
            max_bounds[2],
        ]

    def _calculate_data_center(self):
        """Calculate the center point of all trajectory data"""
        all_points = self.trajectories.reshape(-1, 3)
        return np.mean(all_points, axis=0)

    def _calculate_data_scale(self):
        """Calculate a characteristic scale of the data for camera positioning"""
        bounds = self.data_bounds
        x_range = bounds[1] - bounds[0]
        y_range = bounds[3] - bounds[2]
        z_range = bounds[5] - bounds[4]
        return max(x_range, y_range, z_range)

    def _update_frame(self, frame_idx):
        """Update single frame efficiently for all trajectories"""
        # Calculate trail indices
        start_idx = max(0, frame_idx - self.trail_length + 1)
        end_idx = frame_idx + 1

        # Update each trajectory
        for traj_idx in range(self.n_traj):
            trajectory = self.trajectories[traj_idx, :, :]
            trail_mesh = self.trail_meshes[traj_idx]
            point_mesh = self.point_meshes[traj_idx]

            # Update trail
            if frame_idx > 0:
                trail_points = trajectory[start_idx:end_idx]

                # Create lines for trail
                lines = []
                for i in range(len(trail_points) - 1):
                    lines.extend([2, i, i + 1])

                # Update trail mesh
                trail_mesh.points = trail_points
                trail_mesh.lines = np.array(lines)

            # Update current point
            if frame_idx < len(trajectory):
                current_point = trajectory[frame_idx : frame_idx + 1]
                point_mesh.points = current_point
                # Ensure we have vertex data for point rendering
                if len(current_point) > 0:
                    point_mesh.verts = np.array([1, 0])

    def animate(self, video_path: Path | None = None):
        """Run the animation

        Args:
            video_path (Path | None): Path to save video file. If None, no
                video is saved.
        """
        save_video = video_path is not None

        # Early exit if nothing to do
        if not save_video and self.headless:
            print("Warning: No video output or interactive display - nothing to do")
            return

        # Setup video recording
        if save_video:
            self.plotter.open_movie(str(video_path))

        # Setup display mode
        if not self.headless:
            self.plotter.show(interactive_update=True, auto_close=False)

        # Animation loop
        for frame in range(self.n_timesteps):
            self._update_frame(frame)

            if save_video:
                self.plotter.write_frame()

            if not self.headless:
                self.plotter.update()
                time.sleep(self._delay)  # Only delay for interactive mode

                # Break if window is closed
                if not self.headless and not self.plotter.render_window:
                    break

        self.plotter.close()


def visualize_latent_trajectory(
    latent_space_data: np.ndarray,
    source_data_freq: int,
    play_speed: float = 0.1,
    trail_duration_sec: float = 0.05,
    output_fps: int = 30,
    video_path: Path | None = None,
    headless: bool = True,
):
    """
    Visualize trajectories of a set of variables in a latent space.

    Args:
        latent_space_data (np.ndarray): np.array of shape
            (n_trajectories, n_timesteps, latent_dim).
        source_data_freq (int): Frequency (Hz) of the original source data
            (i.e. along the n_timesteps axis).
        play_speed (float): Fraction of real time to play the animation.
        trail_duration_sec (float): Duration (seconds) of the trail behind
            the current point, counted in output display time (i.e. a 1
            second trail at 0.1x play speed will show 0.1 seconds of data).
        output_fps (int): Frames per second of the output animation.
        video_path (Path | None): Path to save the output video. If None,
            no video is saved.
        headless (bool): Whether to run in offscreen mode (no display).
    """
    # Resample data
    sampling_freq_from_source_data = output_fps / play_speed
    t_source = np.arange(latent_space_data.shape[1]) / source_data_freq
    t_resampled = np.arange(0, t_source[-1], 1.0 / sampling_freq_from_source_data)
    interp_func = interp1d(
        t_source,
        latent_space_data,
        axis=1,
        kind="nearest",
        fill_value="extrapolate",
    )
    latent_space_data_resampled = interp_func(t_resampled)
    latent_dim = latent_space_data_resampled.shape[2]

    # Fit PCA
    pca = PCA(n_components=3)
    pca_x_mat = latent_space_data_resampled.reshape(-1, latent_dim)
    pca.fit(pca_x_mat)
    full_x_mat = latent_space_data.reshape(-1, latent_dim)
    n_variants, n_samples, _ = latent_space_data.shape
    latent_space_data_pca = pca.transform(full_x_mat).reshape(n_variants, n_samples, 3)

    # Visualize
    visualizer = LatentSpaceTrajectoryVisualizer(
        latent_space_data_pca,
        trail_length=int(trail_duration_sec * source_data_freq),
        fps=output_fps,
        headless=headless,
    )
    visualizer.animate(video_path=video_path)
