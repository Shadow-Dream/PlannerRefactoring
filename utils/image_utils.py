"""Image processing utility functions."""
import base64
import numpy as np
import cv2 as cv


def get_base64(image):
    """Convert image to base64 string."""
    _, buffer = cv.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')


def resize_foreground_coord_mask(mask, coords, ratio=0.85):
    """Resize foreground with coordinate mapping."""
    alpha = np.where(mask[..., 0] > 0)
    y1, y2, x1, x2 = (
        alpha[0].min(),
        alpha[0].max(),
        alpha[1].min(),
        alpha[1].max(),
    )

    fg_mask = mask[y1:y2, x1:x2]
    fg_coords = coords[y1:y2, x1:x2]
    size = max(fg_mask.shape[0], fg_mask.shape[1])

    ph0, pw0 = (size - fg_mask.shape[0]) // 2, (size - fg_mask.shape[1]) // 2
    ph1, pw1 = size - fg_mask.shape[0] - ph0, size - fg_mask.shape[1] - pw0
    new_mask = np.pad(
        fg_mask,
        ((ph0, ph1), (pw0, pw1), (0, 0)),
        mode="constant",
        constant_values=((0, 0), (0, 0), (0, 0)),
    )

    new_coords = np.pad(
        fg_coords,
        ((ph0, ph1), (pw0, pw1), (0, 0)),
        mode="constant",
        constant_values=((0, 0), (0, 0), (0, 0)),
    )
    new_size = int(new_mask.shape[0] / ratio)
    ph0, pw0 = (new_size - size) // 2, (new_size - size) // 2
    ph1, pw1 = new_size - size - ph0, new_size - size - pw0
    new_mask = np.pad(
        new_mask,
        ((ph0, ph1), (pw0, pw1), (0, 0)),
        mode="constant",
        constant_values=((0, 0), (0, 0), (0, 0)),
    )
    new_coords = np.pad(
        new_coords,
        ((ph0, ph1), (pw0, pw1), (0, 0)),
        mode="constant",
        constant_values=((0, 0), (0, 0), (0, 0)),
    )
    return new_mask, new_coords


def resize_foreground(image, ratio=0.85):
    """Resize image to fit foreground content."""
    alpha = np.where(image[..., 3] > 0)
    y1, y2, x1, x2 = (
        alpha[0].min(),
        alpha[0].max(),
        alpha[1].min(),
        alpha[1].max(),
    )

    fg = image[y1:y2, x1:x2]
    size = max(fg.shape[0], fg.shape[1])
    ph0, pw0 = (size - fg.shape[0]) // 2, (size - fg.shape[1]) // 2
    ph1, pw1 = size - fg.shape[0] - ph0, size - fg.shape[1] - pw0
    new_image = np.pad(
        fg,
        ((ph0, ph1), (pw0, pw1), (0, 0)),
        mode="constant",
        constant_values=((0, 0), (0, 0), (0, 0)),
    )

    new_size = int(new_image.shape[0] / ratio)
    ph0, pw0 = (new_size - size) // 2, (new_size - size) // 2
    ph1, pw1 = new_size - size - ph0, new_size - size - pw0
    new_image = np.pad(
        new_image,
        ((ph0, ph1), (pw0, pw1), (0, 0)),
        mode="constant",
        constant_values=((0, 0), (0, 0), (0, 0)),
    )
    return new_image


def get_marker(image, last_point, color=(255, 255, 255, 255)):
    """Add a circular marker to the image."""
    x, y = last_point
    image = image + 0
    cv.circle(image, (x, y), 25, (0, 0, 0, 255), -1)
    cv.circle(image, (x, y), 20, color, -1)
    return image


def get_normal_map(point_grid, mask):
    """Generate normal map from point grid."""
    delta_x = point_grid[1:] - point_grid[:-1]
    delta_y = point_grid[:, 1:] - point_grid[:, :-1]
    delta_xl, delta_xr = delta_x[:, :-1], -delta_x[:, 1:]
    delta_yt, delta_yb = delta_y[:-1], -delta_y[1:]
    a = np.cross(delta_xl, delta_yt, axis=-1)
    b = np.cross(delta_yt, delta_xr, axis=-1)
    c = np.cross(delta_xr, delta_yb, axis=-1)
    d = np.cross(delta_yb, delta_xl, axis=-1)
    n = a + b + c + d
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    n = np.abs(n)
    n[~mask[1:, 1:]] = 0
    n = (n * 255).astype(np.uint8)
    normal_map = mask + 0
    normal_map[:-1, :-1] = n
    return normal_map.astype(np.uint8)


def get_height_map(point_grid, mask):
    """Generate height map from point grid."""
    point_grid = point_grid[:, :, 2]
    mask = mask[:, :, 2]
    valid_point = point_grid[mask]
    z_min = valid_point.min()
    z_max = valid_point.max()
    point_grid = (point_grid - z_min) / (z_max - z_min)
    point_grid[~mask] = 0
    point_grid = (point_grid * 255).astype(np.uint8)
    heatmap = cv.applyColorMap(point_grid, cv.COLORMAP_RAINBOW)
    heatmap[~mask] = 0
    return heatmap


def get_mask_bbox(mask, min_margin=10):
    """Get bounding box of non-zero region in mask with minimum margin.

    Args:
        mask: Binary mask array (H, W) or (H, W, C)
        min_margin: Minimum margin in pixels between labels

    Returns:
        tuple: (x1, y1, x2, y2) bounding box coordinates
    """
    if len(mask.shape) == 3:
        mask_2d = mask[..., 0] if mask.shape[2] > 0 else mask
    else:
        mask_2d = mask

    # Find non-zero pixels
    ys, xs = np.where(mask_2d > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    return (x1, y1, x2, y2)


def draw_grid_labels(image, mask, n_grid=8, output_size=512):
    """Draw N*N grid labels on image based on object's bounding box from mask.

    Args:
        image: Input image (H, W, C) with alpha channel
        mask: Binary mask (H, W) or (H, W, C)
        n_grid: Grid size (N*N labels will be placed)
        output_size: Output image size

    Returns:
        tuple: (labeled_image, label_positions)
            - labeled_image: Image with grid labels drawn
            - label_positions: Dict mapping label number to (x, y) pixel coordinates
    """
    # Get mask bounding box
    bbox = get_mask_bbox(mask)
    if bbox is None:
        return image.copy(), {}

    x1, y1, x2, y2 = bbox
    bbox_w = x2 - x1
    bbox_h = y2 - y1

    # Calculate label spacing with padding
    # L/(N+1) spacing means labels are at L/(N+1), 2*L/(N+1), ..., N*L/(N+1)
    step_x = bbox_w / (n_grid + 1)
    step_y = bbox_h / (n_grid + 1)

    # Create output image
    labeled_image = image.copy()
    if len(labeled_image.shape) == 2:
        labeled_image = cv.cvtColor(labeled_image, cv.COLOR_GRAY2BGR)
    elif labeled_image.shape[2] == 4:
        # Keep RGBA
        pass

    label_positions = {}

    # Calculate font scale based on spacing (smaller font to preserve image info)
    min_spacing = min(step_x, step_y)
    font_scale = max(0.25, min(0.5, min_spacing / 80))
    thickness = max(1, int(font_scale * 2))

    # Draw labels in grid
    for row in range(n_grid):
        for col in range(n_grid):
            label_num = row * n_grid + col + 1  # 1-indexed

            # Calculate position with padding
            px = int(x1 + (col + 1) * step_x)
            py = int(y1 + (row + 1) * step_y)

            # Store position
            label_positions[label_num] = (px, py)

            # Get text size for centering
            text = str(label_num)
            (text_w, text_h), baseline = cv.getTextSize(
                text, cv.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )

            # Center text on position
            text_x = px - text_w // 2
            text_y = py + text_h // 2

            # Draw black outline (stroke)
            outline_thickness = thickness + 2
            cv.putText(
                labeled_image, text, (text_x, text_y),
                cv.FONT_HERSHEY_SIMPLEX, font_scale,
                (0, 0, 0, 255) if labeled_image.shape[2] == 4 else (0, 0, 0),
                outline_thickness, cv.LINE_AA
            )

            # Draw white fill
            cv.putText(
                labeled_image, text, (text_x, text_y),
                cv.FONT_HERSHEY_SIMPLEX, font_scale,
                (255, 255, 255, 255) if labeled_image.shape[2] == 4 else (255, 255, 255),
                thickness, cv.LINE_AA
            )

    return labeled_image, label_positions


def draw_grid_labels_on_object(image, mask, point_grid, n_grid=8, output_path=None,
                               target_ratio=0.9, output_size=512):
    """Draw grid labels on object image with auto-scaling for better visibility.

    This function scales the object to fill more of the image if it's too small,
    then draws N*N numbered labels on the object. The 3D coordinates are properly
    mapped back to the original coordinate system.

    Args:
        image: Input image (H, W, C) with alpha channel
        mask: Binary mask (H, W) or (H, W, C)
        point_grid: 3D coordinate grid (H, W, 3)
        n_grid: Grid size (N*N labels)
        output_path: Optional path to save debug image
        target_ratio: Target ratio of bbox longest axis to image size (default 0.9)
        output_size: Output image size (default 512)

    Returns:
        tuple: (labeled_image, label_to_coord)
            - labeled_image: Image with grid labels (scaled to output_size)
            - label_to_coord: Dict mapping label number to 3D coordinates
    """
    # Get original dimensions
    orig_h, orig_w = image.shape[:2]

    # Get mask bounding box
    bbox = get_mask_bbox(mask)
    if bbox is None:
        return image.copy(), {}

    x1, y1, x2, y2 = bbox
    bbox_w = x2 - x1
    bbox_h = y2 - y1
    longest_axis = max(bbox_w, bbox_h)

    # Calculate scale factor if object is too small
    current_ratio = longest_axis / max(orig_h, orig_w)
    if current_ratio < target_ratio:
        scale_factor = target_ratio / current_ratio
    else:
        scale_factor = 1.0

    # Calculate crop and scale parameters
    # Center the bbox in the output image
    bbox_cx = (x1 + x2) / 2
    bbox_cy = (y1 + y2) / 2

    # Size of the view window in original coordinates
    view_size = longest_axis / target_ratio
    view_x1 = bbox_cx - view_size / 2
    view_y1 = bbox_cy - view_size / 2
    view_x2 = bbox_cx + view_size / 2
    view_y2 = bbox_cy + view_size / 2

    # Clamp to image bounds and adjust
    if view_x1 < 0:
        view_x2 -= view_x1
        view_x1 = 0
    if view_y1 < 0:
        view_y2 -= view_y1
        view_y1 = 0
    if view_x2 > orig_w:
        view_x1 -= (view_x2 - orig_w)
        view_x2 = orig_w
    if view_y2 > orig_h:
        view_y1 -= (view_y2 - orig_h)
        view_y2 = orig_h

    # Ensure within bounds
    view_x1 = max(0, view_x1)
    view_y1 = max(0, view_y1)
    view_x2 = min(orig_w, view_x2)
    view_y2 = min(orig_h, view_y2)

    # Convert to integers
    view_x1, view_y1 = int(view_x1), int(view_y1)
    view_x2, view_y2 = int(view_x2), int(view_y2)

    # Crop and resize image
    cropped_image = image[view_y1:view_y2, view_x1:view_x2]
    cropped_mask = mask[view_y1:view_y2, view_x1:view_x2]

    # Resize to output size
    scaled_image = cv.resize(cropped_image, (output_size, output_size),
                             interpolation=cv.INTER_LINEAR)
    scaled_mask = cv.resize(cropped_mask, (output_size, output_size),
                            interpolation=cv.INTER_NEAREST)

    # Draw labels on scaled image
    labeled_image, label_positions = draw_grid_labels(scaled_image, scaled_mask, n_grid)

    # Map label positions back to original coordinates and get 3D coords
    label_to_coord = {}
    grid_h, grid_w = point_grid.shape[:2]

    # Get mask for finding nearest valid point (in original coordinates)
    if len(mask.shape) == 3:
        mask_2d = mask[..., 0] > 0
    else:
        mask_2d = mask > 0

    mask_coords = np.stack(np.where(mask_2d)).T  # (N, 2) in (y, x) format

    # Calculate scale from output back to crop, then to original
    crop_h = view_y2 - view_y1
    crop_w = view_x2 - view_x1

    for label_num, (px, py) in label_positions.items():
        # Convert from scaled image coords to cropped image coords
        crop_x = px * crop_w / output_size
        crop_y = py * crop_h / output_size

        # Convert from cropped coords to original image coords
        orig_x = crop_x + view_x1
        orig_y = crop_y + view_y1

        # Convert to point_grid coordinates
        gx = int(orig_x * grid_w / orig_w)
        gy = int(orig_y * grid_h / orig_h)

        # Clamp to valid range
        gx = max(0, min(grid_w - 1, gx))
        gy = max(0, min(grid_h - 1, gy))

        # Find nearest valid point in mask
        if mask_2d[gy, gx]:
            coord_3d = point_grid[gy, gx]
        else:
            # Find nearest masked point
            if len(mask_coords) > 0:
                distances = np.linalg.norm(mask_coords - np.array([gy, gx]), axis=1)
                nearest_idx = distances.argmin()
                ny, nx = mask_coords[nearest_idx]
                coord_3d = point_grid[ny, nx]
            else:
                coord_3d = np.array([0, 0, 0])

        label_to_coord[label_num] = coord_3d.tolist()

    # Save debug image if path provided
    if output_path:
        # Convert RGBA to BGR for saving
        if labeled_image.shape[2] == 4:
            save_image = cv.cvtColor(labeled_image, cv.COLOR_RGBA2BGRA)
        else:
            save_image = labeled_image
        cv.imwrite(output_path, save_image)

    return labeled_image, label_to_coord
