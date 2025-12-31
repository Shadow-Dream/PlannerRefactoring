"""Geometry calculation utility functions."""
import numpy as np
import shapely as sl
from scipy.spatial import ConvexHull


def get_convex_hull(points):
    """Compute convex hull polygon from points."""
    try:
        hull = ConvexHull(points)
        contour = sl.Polygon(points[hull.vertices])
        return contour
    except:
        return None


def polygon_to_list(polygon):
    """Convert shapely polygon to numpy array."""
    x, y = polygon.exterior.xy
    polygon = np.array([x, y], dtype=np.float32).T
    return polygon


def linestring_to_list(linestring):
    """Convert shapely linestring to numpy array."""
    x, y = linestring.xy
    linestring = np.array([x, y], dtype=np.float32).T
    return linestring


def get_instance_distance(a, b):
    """Calculate distance between two object instances based on their bounding boxes."""
    abbox = np.array(a["bbox"], dtype=np.float32) + np.array(a["position"], dtype=np.float32)
    bbbox = np.array(b["bbox"], dtype=np.float32) + np.array(b["position"], dtype=np.float32)
    abbox = abbox[::2, :2]
    bbbox = bbbox[::2, :2]
    abbox = sl.Polygon(abbox)
    bbbox = sl.Polygon(bbbox)
    distance = abbox.distance(bbbox)
    return distance


def get_instance_position_distance(a, b):
    """Calculate distance between an object instance and a position."""
    bbox = np.array(a["bbox"], dtype=np.float32) + np.array(a["position"], dtype=np.float32)
    bbox = bbox[::2, :2]
    bbox = sl.Polygon(bbox)
    position = sl.Point(np.array(b, dtype=np.float32)[:2])
    distance = bbox.distance(position)
    return distance
