import numpy as np
from scipy.spatial.transform import Rotation


class CameraToWorldMapper:
    """Map 2D camera coordinates + depth to 3D world coordinates given
    camera parameters."""

    def __init__(
        self,
        camera_pos: tuple[float, float, float],
        camera_fov_deg: float,
        rendering_size: tuple[int, int],
        rotation_euler: tuple[float, float, float],
    ):
        """Initialize the CameraToWorldMapper with camera parameters. This
        basically tries to reproduce the camera model in dm_control (see
        https://github.com/google-deepmind/dm_control/blob/9e3d96c1dd986aab52953f0bfd6d968e7b9b7a6a/dm_control/mujoco/engine.py#L737-L786)

        Note that MuJoCo uses the following conventions:
            "Cameras look towards the negative Z axis of the camera frame,
            while positive X and Y correspond to right and up in the image
            plane, respectively."
        (see https://mujoco.readthedocs.io/en/stable/modeling.html#cameras)

        Args:
            camera_pos: (x, y, z) position of the camera in world coords
                (in mm).
            camera_fov_deg: Camera field of view in degrees (assumed
                it is the same horizontally and vertically).
            rendering_size: (width, height) of the rendered image in
                pixels. Must be square (width == height).
            rotation_euler: (roll, pitch, yaw) rotation of the camera in
                radians. These should be the exact same values that were
                used to create the camera during simulation.
        """
        # Check inputs
        camera_pos = np.array(camera_pos)
        if len(camera_pos) != 3:
            raise ValueError("camera_pos should be a 3-element sequence")

        if len(rendering_size) != 2 or rendering_size[0] != rendering_size[1]:
            raise NotImplementedError("Only square rendering sizes are supported")
        img_size = rendering_size[0]

        rotation_euler = np.array(rotation_euler)
        if len(rotation_euler) != 3:
            raise ValueError("rotation_euler should be a 3-element sequence")

        self.camera_pos = camera_pos
        self.camera_fov_deg = camera_fov_deg
        self.rendering_size = rendering_size
        self.rotation_euler = rotation_euler

        # Image matrix (3x3)
        self.image_mat = np.eye(3)
        self.image_mat[0, 2] = (img_size - 1) / 2.0
        self.image_mat[1, 2] = (img_size - 1) / 2.0

        # Focal transformation matrix (3x4)
        obj_size = img_size / 2  # side opposite
        angle_rad = np.deg2rad(camera_fov_deg / 2)  # angle
        focal_scaling = obj_size / np.tan(angle_rad)  # side adjacent
        # focal_transform_mat[0, 0] should have a sign flip because the camera looks at
        # the NEGATIVE z axis (not entirely sure how this works out, but this is what
        # dm_control does within its Camera implementation)
        self.focal_transform_mat = np.diag([-focal_scaling, focal_scaling, 1, 0])[:3, :]

        # Camera rotation matrix (4x4)
        cam_rotation = Rotation.from_euler("xyz", rotation_euler, degrees=False)
        self.cam_rotation_mat = np.eye(4)
        self.cam_rotation_mat[:3, :3] = cam_rotation.as_matrix()

        # Translation matrix (4x4)
        self.translation_mat = np.eye(4)
        self.translation_mat[:3, 3] = -camera_pos

        # Overall camera matrix (3x4)
        self.camera_matrix = (
            self.image_mat
            @ self.focal_transform_mat
            @ self.cam_rotation_mat
            @ self.translation_mat
        )

        # Compute 3x3 camera intrinsic matrix - this is the K in P = K[R|t]
        # (only need the first 3 columns because the last column is all 0s)
        self.camera_intrinsic_mat = self.image_mat @ self.focal_transform_mat[:, :3]
        self.camera_intrinsic_mat_inv = np.linalg.inv(self.camera_intrinsic_mat)

        # Compute the 4x4 camera extrinsic matrix - this is the [R|t] in P = K[R|t]
        self.camera_extrinsic_mat = self.cam_rotation_mat @ self.translation_mat
        self.camera_extrinsic_mat_inv = np.linalg.inv(self.camera_extrinsic_mat)

    def __call__(self, xy, depth):
        """Map 2D camera coordinates + depth to 3D world coordinates.

        Args:
            xy: (..., 2) array of 2D camera coordinates (in pixels).
            depth: (...,) array of depth values (in mm).

        Returns:
            (..., 3) array of 3D world coordinates (in mm).
        """
        xy_shape = xy.shape
        depth_shape = depth.shape
        if len(xy_shape) < 2 or xy_shape[-1] != 2 or xy_shape[:-1] != depth_shape:
            raise ValueError(
                "xy and depth shapes are incompatible: "
                "xy should have shape (*depth.shape, 2), "
                f"got xy.shape {xy_shape} and depth.shape {depth_shape}"
            )

        # Convert to projections of shape (..., 3) where the columns are (x, y, depth)
        projection = np.hstack((xy.reshape(-1, 2), depth.reshape(-1, 1)))
        n_points_total = projection.shape[0]

        # Unnormalize x, y projections
        # (they used to be [x*depth, y*depth, depth] before perspective division)
        projection[:, 0] *= projection[:, 2]
        projection[:, 1] *= projection[:, 2]

        # Get 3D camera frame coords by inverting the camera intrinsics, i.e. K
        pos_cam = (self.camera_intrinsic_mat_inv @ projection.T).T  # (N, 3)

        # Make camera coords homogeneous (N, 4)
        pos_cam_hom = np.hstack((pos_cam, np.ones((n_points_total, 1))))

        # Invert the extrinsics (rotation & translation, i.e. [R|t])
        pos_world_hom = (self.camera_extrinsic_mat_inv @ pos_cam_hom.T).T  # (N, 4)

        # Reshape back to original shape
        pos_world = pos_world_hom[:, :3].reshape(*xy_shape[:-1], 3)

        return pos_world
