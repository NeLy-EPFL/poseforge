import numpy as np
from scipy import ndimage


def rotate_points_to_align(pts, pts_top, pts_bottom):
    """
    Rotate all points around pts_bottom such that pts_top is directly above
    pts_bottom.

    Args:
        pts: numpy array of shape (num_pts, 2) containing all points
        pts_top: numpy array of shape (2,) - the point to align vertically
        pts_bottom: numpy array of shape (2,) - the center of rotation

    Returns:
        rotated_pts: numpy array of shape (num_pts, 2) with rotated points
        rotation_angle: float, the rotation angle applied in radians
    """
    # Calculate the current vector from bottom to top
    vector = pts_top - pts_bottom

    # Calculate the angle of this vector from the positive x-axis
    current_angle = np.arctan2(vector[1], vector[0])

    # In image coordinates, "up" means negative y direction
    # So target angle is -π/2 (pointing up in image coordinates)
    target_angle = -np.pi / 2

    # Calculate rotation angle needed (counterclockwise)
    rotation_angle = target_angle - current_angle

    # Create rotation matrix for counterclockwise rotation
    cos_theta = np.cos(rotation_angle)
    sin_theta = np.sin(rotation_angle)
    rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

    # Translate points so that pts_bottom is at origin
    translated_pts = pts - pts_bottom

    # Apply rotation
    rotated_translated_pts = translated_pts @ rotation_matrix.T

    # Translate back
    rotated_pts = rotated_translated_pts + pts_bottom

    return rotated_pts, rotation_angle


def rotate_image_around_point(image, rotation_point, rotation_angle, fill_value=0):
    """
    Rotate a grayscale image around a specific point.

    Args:
        image: numpy array of shape (nrows, ncols) - grayscale image
        rotation_point: numpy array of shape (2,) - point to rotate around
            in (x, y) coordinates
        rotation_angle: rotation angle in radians (counterclockwise)
        fill_value: float value to fill areas outside the original image

    Returns:
        rotated_image: numpy array of same shape as input image
    """
    # Get image dimensions
    height, width = image.shape

    # Convert rotation point from (x, y) to (col, row) for matrix operations
    cx, cy = rotation_point[0], rotation_point[1]

    # Create rotation matrix (counterclockwise)
    cos_theta = np.cos(rotation_angle)
    sin_theta = np.sin(rotation_angle)

    # Affine transformation matrix for rotation around point (cx, cy)
    # First translate to origin, then rotate, then translate back
    # But ndimage.affine_transform expects the inverse transformation

    # Forward transformation matrix (what we want to do to points):
    # 1. Translate by (-cx, -cy)
    # 2. Rotate
    # 3. Translate by (cx, cy)

    # For the inverse (what ndimage needs), we do the reverse:
    # 1. Translate by (-cx, -cy)
    # 2. Rotate by -angle
    # 3. Translate by (cx, cy)

    # Inverse rotation matrix
    inv_cos = cos_theta  # cos(-theta) = cos(theta)
    inv_sin = -sin_theta  # sin(-theta) = -sin(theta)

    # Build the inverse transformation matrix for ndimage.affine_transform
    # Note: ndimage works in (row, col) = (y, x) coordinates
    matrix = np.array([[inv_cos, inv_sin], [-inv_sin, inv_cos]])

    # Offset calculation for rotation around (cx, cy)
    # In (y, x) coordinates: rotation point is (cy, cx)
    offset_y = cy - (matrix[0, 0] * cy + matrix[0, 1] * cx)
    offset_x = cx - (matrix[1, 0] * cy + matrix[1, 1] * cx)
    offset = [offset_y, offset_x]

    # Apply the transformation
    rotated_image = ndimage.affine_transform(
        image,
        matrix,
        offset=offset,
        output_shape=image.shape,
        cval=fill_value,
        prefilter=False,
    )

    return rotated_image


def crop_image_and_keypoints(
    image, keypoints, center, crop_dim, x_offset=0, y_offset=0
):
    """
    Crop a square region of size crop_dim x crop_dim centered at 'center'
    (optionally shifted by x_offset, y_offset) from 'image', and shift
    'keypoints' accordingly.

    Args:
        image: numpy array, shape (H, W)
        keypoints: numpy array, shape (N, 2)
        center: tuple or array-like, (x, y) center of crop
        crop_dim: int, size of the square crop
        x_offset: int, shift in x (columns) from center
        y_offset: int, shift in y (rows) from center

    Returns:
        cropped_image: numpy array, shape (crop_dim, crop_dim)
        shifted_keypoints: numpy array, shape (N, 2)
        If crop is out of bounds, returns None
    """
    x_center, y_center = center
    x_center += x_offset
    y_center += y_offset
    x_min = int(np.round(x_center - crop_dim // 2))
    y_min = int(np.round(y_center - crop_dim // 2))
    x_max = x_min + crop_dim
    y_max = y_min + crop_dim

    # If crop is out of bounds, return None
    if x_min < 0 or y_min < 0 or x_max > image.shape[1] or y_max > image.shape[0]:
        return None

    cropped = image[y_min:y_max, x_min:x_max]
    shifted_keypoints = keypoints - np.array([x_min, y_min])

    return cropped, shifted_keypoints
