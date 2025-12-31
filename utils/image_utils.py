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
