"""Navigation backend for path planning."""
import os
import time
import json
import heapq
import numpy as np
import trimesh
import torch
import cv2 as cv
import pyclipper as pc
import shapely as sl
from shapely import ops
import traceback

from planner.core import quaternion
from planner.utils.geometry_utils import get_convex_hull, polygon_to_list, linestring_to_list


class NavigationBackend:
    """Backend process for navigation and path planning."""

    # Configuration constants
    BORDER = 20
    RADIUS = 0.1
    TOLERANCE = 0.2
    POLYGON_RESOLUTION = 100
    RESOLUTION = 512
    SIMPLIFY_TOLERANCE = 0.7

    def __init__(self, env_id, request_queue, result_queue):
        """Initialize navigation backend."""
        self.env_id = env_id
        self.request_queue = request_queue
        self.result_queue = result_queue
        self.capture_dir = f"planner/capture"

    def run(self):
        """Main navigation loop."""
        try:
            floor_poly, floor_poly_no_buffer = self._load_floor()

            while True:
                if self.request_queue.empty():
                    time.sleep(0.1)
                    continue

                request = self.request_queue.get()
                objects, parents, at_position, to_position, to_direction, target = request
                parent = parents[target]

                path = self._plan_path(
                    objects, parents, at_position, to_position,
                    to_direction, target, parent, floor_poly, floor_poly_no_buffer
                )
                self.result_queue.put(path)

        except Exception as e:
            print("发生异常：", e)
            traceback.print_exc()
            self.result_queue.put({"type": "fail"})

    def _load_floor(self):
        """Load floor geometry from scene assets."""
        metadata_path = "planner/scene/config.json"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        room_type = metadata["room"]
        if room_type == "bedroom":
            floor_path = "planner/scene/bedroom_0_0_floor/bedroom_0_0_floor.obj"
        else:
            floor_path = "planner/scene/living-room_0_0_floor/living-room_0_0_floor.obj"

        floor = trimesh.load(floor_path)
        vertices = floor.vertices
        start_point = np.array(torch.load(f"planner/capture/start_point.pt", weights_only=False))

        for key, value in metadata["objects"].items():
            if room_type in key and "floor" in key:
                break
        floor_position = np.array(value["position"])

        vertices = vertices + floor_position - start_point
        vertices = vertices[:, :2]
        faces = floor.faces

        floor_polygon = None
        for face in faces:
            subpoly = vertices[face]
            if floor_polygon is None:
                floor_polygon = sl.Polygon(subpoly).buffer(0)
            else:
                floor_polygon = floor_polygon.union(sl.Polygon(subpoly).buffer(0))
                floor_polygon = floor_polygon.buffer(0)

        floor_poly_no_buffer = floor_polygon
        floor_poly = floor_polygon.buffer(-self.RADIUS, join_style="mitre")

        return floor_poly, floor_poly_no_buffer

    def _build_scene_polygons(self, objects, parents):
        """Build scene polygons from objects."""
        main_objects = list(set(parents.values()))
        objects = {
            name: instance for name, instance in objects.items()
            if name in main_objects
        }

        inner_polygon = None
        inner_polygon_no_buffer = None
        object_polygons = {}

        for name, instance in objects.items():
            position = np.array(instance["position"])
            rotation = np.array(instance["rotation"])
            shape = instance["shape"]
            quat = rotation[None][None]
            quat = np.broadcast_to(quat, (shape.shape[0], shape.shape[1], 4))
            shape = quaternion.qrot_np(quat, shape)
            shape += position[None][None]
            shape = shape[:, :, :2]

            object_polygon = None
            for box in shape:
                polygon = get_convex_hull(box)
                if polygon is None:
                    continue
                if object_polygon is None:
                    object_polygon = polygon
                else:
                    object_polygon = object_polygon.union(polygon)

            # Skip objects with invalid/empty shapes
            if object_polygon is None:
                continue

            object_polygon_no_buffer = object_polygon
            object_polygon = object_polygon.buffer(self.RADIUS, join_style="mitre")
            object_polygons[name] = object_polygon

            if inner_polygon is None:
                inner_polygon = object_polygon
                inner_polygon_no_buffer = object_polygon_no_buffer
            else:
                inner_polygon = inner_polygon.union(object_polygon)
                inner_polygon_no_buffer = inner_polygon_no_buffer.union(object_polygon_no_buffer)

        return inner_polygon, inner_polygon_no_buffer, object_polygons

    def _clip_polygons(self, inner_polygon, floor_poly):
        """Clip inner polygons with floor polygon."""
        def to_pixel(points):
            return (points * self.POLYGON_RESOLUTION).astype(np.int32)

        if isinstance(inner_polygon, sl.Polygon):
            polygons = [polygon_to_list(inner_polygon)]
        else:
            polygons = [polygon_to_list(polygon) for polygon in inner_polygon.geoms]

        if isinstance(floor_poly, sl.MultiPolygon):
            largest_subpoly = None
            for subpoly in floor_poly.geoms:
                if largest_subpoly is None or largest_subpoly.area < subpoly.area:
                    largest_subpoly = subpoly
            floor_poly = largest_subpoly
        else:
            floor_poly = floor_poly.buffer(0)

        floor = polygon_to_list(floor_poly)
        polygons = [to_pixel(polygon) for polygon in polygons]
        floor = to_pixel(floor)

        clipper = pc.Pyclipper()
        clipper.AddPath(floor, pc.PT_SUBJECT, True)
        clipper.AddPaths(polygons, pc.PT_CLIP, True)
        polygons = clipper.Execute(pc.CT_INTERSECTION, pc.PFT_EVENODD, pc.PFT_EVENODD)
        clipper.Clear()
        clipper.AddPath(floor, pc.PT_SUBJECT, True)
        clipper.AddPaths(polygons, pc.PT_CLIP, True)
        floor = clipper.Execute(pc.CT_DIFFERENCE, pc.PFT_EVENODD, pc.PFT_EVENODD)

        polygons = [
            np.array(polygon, dtype=np.float32) / self.POLYGON_RESOLUTION
            for polygon in polygons
        ]
        floor_polygons = [
            np.array(polygon, dtype=np.float32) / self.POLYGON_RESOLUTION
            for polygon in floor
        ]

        inner_polygon_result = None
        for polygon in polygons:
            if inner_polygon_result is None:
                inner_polygon_result = sl.Polygon(polygon).buffer(0)
            else:
                inner_polygon_result = inner_polygon_result.union(sl.Polygon(polygon).buffer(0))
                inner_polygon_result = inner_polygon_result.buffer(0)

        floor_result = None
        for polygon in floor_polygons:
            if floor_result is None:
                floor_result = sl.Polygon(polygon).buffer(0)
            else:
                floor_result = floor_result.union(sl.Polygon(polygon).buffer(0))
                floor_result = floor_result.buffer(0)

        return inner_polygon_result, floor_result

    def _compute_positions(self, at_position, to_position, to_direction, parent,
                          inner_polygon, floor, object_polygons):
        """Compute start and end positions for path planning."""
        at_position = np.array(at_position, dtype=np.float32)
        to_position = np.array(to_position, dtype=np.float32)
        at_point = sl.Point(at_position)
        to_point = sl.Point(to_position)
        boundary = None

        # Compute final at_position
        if not floor.contains(at_point):
            nearest = ops.nearest_points(floor.boundary, at_point)
            final_at_position = [nearest[0].x, nearest[0].y]
        elif inner_polygon.contains(at_point):
            nearest = ops.nearest_points(inner_polygon.boundary, at_point)
            final_at_position = [nearest[0].x, nearest[0].y]
        else:
            final_at_position = at_position
        final_at_position = np.array(final_at_position, dtype=np.float32)

        # Compute final to_position
        if not floor.contains(to_point):
            contain_polygon = object_polygons[parent]
            boundary = contain_polygon.buffer(0.01, join_style="mitre").intersection(floor).boundary

            if to_direction[0] == 0 and to_direction[1] == 0:
                to_direction = np.array(at_position, dtype=np.float32) - np.array(to_position, dtype=np.float32)
            to_direction = to_direction / np.linalg.norm(to_direction)
            to_direction = to_direction * 10 + to_position
            to_direction = sl.Point(to_direction)
            ray = sl.LineString([to_point, to_direction])
            intersection = boundary.intersection(ray)

            if intersection:
                if isinstance(intersection, sl.MultiPoint):
                    min_distance = 10
                    for intersection_candidate in intersection.geoms:
                        cur_distance = sl.distance(to_point, intersection_candidate)
                        if cur_distance < min_distance:
                            min_distance = cur_distance
                            intersection = intersection_candidate
                final_to_position = [intersection.x, intersection.y]
            else:
                intersection = ops.nearest_points(boundary, ray)[0]
                final_to_position = [intersection.x, intersection.y]

        elif inner_polygon.contains(to_point):
            if to_direction[0] == 0 and to_direction[1] == 0:
                to_direction = np.array(at_position, dtype=np.float32) - np.array(to_position, dtype=np.float32)
            to_direction = to_direction / np.linalg.norm(to_direction)
            to_direction = to_direction * 10 + to_position
            to_direction = sl.Point(to_direction)
            ray = sl.LineString([to_point, to_direction])
            intersection = inner_polygon.boundary.intersection(ray)

            if isinstance(intersection, sl.MultiPoint):
                min_distance = 10
                for intersection_candidate in intersection.geoms:
                    cur_distance = sl.distance(to_point, intersection_candidate)
                    if cur_distance < min_distance:
                        min_distance = cur_distance
                        intersection = intersection_candidate
            final_to_position = [intersection.x, intersection.y]
        else:
            final_to_position = to_position
        final_to_position = np.array(final_to_position, dtype=np.float32)

        return final_at_position, final_to_position, boundary

    def _build_bitmap(self, floor, inner_polygon, inner_polygon_no_buffer,
                     floor_poly_no_buffer, min_x, min_y, max_axis):
        """Build bitmap for path planning."""
        def points_to_pixel(points):
            points = (points - np.array([min_x, min_y], dtype=np.float32)) / max_axis * self.RESOLUTION
            return points.astype(np.int32)

        bitmap = np.zeros((self.RESOLUTION, self.RESOLUTION), dtype=np.uint8)
        bitmap_no_buffer = np.zeros((self.RESOLUTION, self.RESOLUTION), dtype=np.uint8)
        kernel = np.ones((5, 5), np.uint8)

        # Draw floor
        if isinstance(floor, sl.Polygon):
            points = points_to_pixel(polygon_to_list(floor))
            cv.fillPoly(bitmap, [points], 255)
        else:
            for polygon in floor.geoms:
                points = points_to_pixel(polygon_to_list(polygon))
                cv.fillPoly(bitmap, [points], 255)

        if isinstance(floor_poly_no_buffer, sl.Polygon):
            points = points_to_pixel(polygon_to_list(floor_poly_no_buffer))
            cv.fillPoly(bitmap_no_buffer, [points], 255)
        else:
            for polygon in floor_poly_no_buffer.geoms:
                points = points_to_pixel(polygon_to_list(polygon))
                cv.fillPoly(bitmap_no_buffer, [points], 255)

        # Remove inner polygon (obstacles)
        if isinstance(inner_polygon, sl.Polygon):
            points = points_to_pixel(polygon_to_list(inner_polygon))
            cv.fillPoly(bitmap, [points], 0)
        else:
            for polygon in inner_polygon.geoms:
                points = points_to_pixel(polygon_to_list(polygon))
                cv.fillPoly(bitmap, [points], 0)

        if isinstance(inner_polygon_no_buffer, sl.Polygon):
            points = points_to_pixel(polygon_to_list(inner_polygon_no_buffer))
            cv.fillPoly(bitmap_no_buffer, [points], 127)
        else:
            for polygon in inner_polygon_no_buffer.geoms:
                points = points_to_pixel(polygon_to_list(polygon))
                cv.fillPoly(bitmap_no_buffer, [points], 127)

        cv.imwrite(f"{self.capture_dir}/bev_map.png", bitmap_no_buffer)
        torch.save((min_x, min_y, max_axis, self.RESOLUTION), f"{self.capture_dir}/bev_map_state.pt")

        # Morphological operations
        bitmap = cv.morphologyEx(bitmap, cv.MORPH_OPEN, kernel)
        bitmap = cv.morphologyEx(bitmap, cv.MORPH_CLOSE, kernel)
        cv.imwrite(f"{self.capture_dir}/bev_map_buffer.png", bitmap)

        # Distance transform
        final_bitmap = np.zeros_like(bitmap)
        current_bitmap = bitmap.copy()
        i = 0
        while (current_bitmap > 0).any():
            i += 1
            current_bitmap = cv.erode(current_bitmap, kernel, borderType=cv.BORDER_CONSTANT, borderValue=0)
            final_bitmap[(current_bitmap == 0) & (bitmap > 0)] = i
            bitmap = current_bitmap

        return final_bitmap.T

    def _astar(self, matrix, start, end, max_axis, offwall_ratio=0.1):
        """A* path planning algorithm."""
        mask = (matrix == 0)
        matrix = matrix.astype(np.float32)
        pixel_margin = int(0.5 / max_axis * self.RESOLUTION + 1)
        matrix /= pixel_margin
        matrix = 1 - matrix
        matrix = np.clip(matrix, 0.01, 1)
        matrix = (1 - offwall_ratio) * matrix + offwall_ratio
        matrix[mask] = 0

        start = tuple(start)
        end = tuple(end)
        n = len(matrix)
        open_list = []
        heapq.heappush(open_list, (0, start[0], start[1]))
        came_from = {}
        g_score = {}
        g_score[start] = 0

        while open_list:
            current_f, x, y = heapq.heappop(open_list)
            if (x, y) == end:
                return self._reconstruct_path(came_from, end)

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < n and 0 <= ny < n and matrix[nx, ny] != 0:
                    heuristic = (abs(nx - end[0]) + abs(ny - end[1])) * offwall_ratio
                    tentative_g = g_score[(x, y)] + matrix[x, y]
                    f = tentative_g + heuristic

                    if tentative_g < g_score.get((nx, ny), float('inf')):
                        came_from[(nx, ny)] = (x, y)
                        g_score[(nx, ny)] = tentative_g
                        heapq.heappush(open_list, (f, nx, ny))
        return None

    def _reconstruct_path(self, came_from, current):
        """Reconstruct path from A* result."""
        path = []
        while current in came_from:
            path.append(current)
            current = came_from[current]
        path.reverse()
        return path

    def _simplify_path(self, path, floor, inner_polygon):
        """Simplify path by removing unnecessary waypoints."""
        sim_start_index = 0
        while sim_start_index < len(path) - 1:
            sim_end_index = sim_start_index + 2
            while sim_end_index < len(path):
                segment = sl.LineString([path[sim_start_index], path[sim_end_index]])
                intersection = floor.intersection(segment)
                if intersection != segment:
                    break
                distance = floor.boundary.distance(segment)
                if distance < self.SIMPLIFY_TOLERANCE:
                    break
                sim_end_index += 1

                intersection = inner_polygon.intersection(segment)
                if not intersection.is_empty:
                    break
                distance = inner_polygon.distance(segment)
                if distance < self.SIMPLIFY_TOLERANCE:
                    break

            path = path[:sim_start_index + 1] + path[sim_end_index - 1:]
            sim_start_index += 1
        return path

    def _plan_path(self, objects, parents, at_position, to_position,
                  to_direction, target, parent, floor_poly, floor_poly_no_buffer):
        """Plan path from current position to target."""
        # Build scene
        inner_polygon, inner_polygon_no_buffer, object_polygons = self._build_scene_polygons(
            objects, parents
        )
        inner_polygon, floor = self._clip_polygons(inner_polygon, floor_poly)

        # Compute positions
        final_at_position, final_to_position, boundary = self._compute_positions(
            at_position, to_position, to_direction, parent,
            inner_polygon, floor, object_polygons
        )

        # Compute bitmap bounds
        min_x, min_y, max_x, max_y = floor.bounds
        max_axis = max(max_x - min_x, max_y - min_y)
        center_x = (max_x + min_x) / 2
        center_y = (max_y + min_y) / 2
        min_x = center_x - max_axis / 2
        min_y = center_y - max_axis / 2

        def points_to_pixel(points):
            points = (points - np.array([min_x, min_y], dtype=np.float32)) / max_axis * self.RESOLUTION
            return points.astype(np.int32)

        def pixel_to_points(points):
            points = points.astype(np.float32)
            points = points / self.RESOLUTION * max_axis
            points = points + np.array([min_x, min_y], dtype=np.float32)
            return points

        # Build bitmap
        bitmap = self._build_bitmap(
            floor, inner_polygon, inner_polygon_no_buffer,
            floor_poly_no_buffer, min_x, min_y, max_axis
        )

        # Find valid start and end positions
        coords = np.argwhere(bitmap > 0)
        pixel_at_position = points_to_pixel(final_at_position)
        at_distance = coords - pixel_at_position
        at_distance = np.linalg.norm(at_distance, axis=-1)
        min_index = at_distance.argmin()
        at_position_pixel = coords[min_index]

        pixel_to_position = points_to_pixel(final_to_position)
        to_distance = coords - pixel_to_position
        to_distance = np.linalg.norm(to_distance, axis=-1)
        min_index = to_distance.argmin()
        to_position_pixel = coords[min_index]

        # Run A*
        path = self._astar(bitmap, at_position_pixel, to_position_pixel, max_axis)
        if path is None or len(path) < 2:
            return []

        path = np.array(path)

        # Visualization
        image = cv.cvtColor(bitmap, cv.COLOR_GRAY2BGR)
        image[image > 0] = 255
        for point in path:
            image[point[0], point[1]] = [0, 0, 255]
        cv.circle(image, tuple(pixel_at_position[::-1]), 5, (0, 255, 0), -1)
        cv.circle(image, tuple(pixel_to_position[::-1]), 5, (0, 255, 0), -1)

        # Convert and simplify path
        path = pixel_to_points(path)
        path = sl.LineString(path)
        path = path.simplify(0.1)
        path = path.coords._coords
        path = np.array(path, dtype=np.float32).tolist()

        # Simplify path further
        path = self._simplify_path(path, floor, inner_polygon)
        path = points_to_pixel(np.array(path))

        # Draw path on image
        for i in range(len(path) - 1):
            cv.circle(image, tuple(path[i][::-1]), 3, (255, 0, 0), -1)
            cv.line(
                image,
                tuple(points_to_pixel(path[i][::-1])),
                tuple(points_to_pixel(path[i + 1][::-1])),
                (0, 255, 0), 2
            )

        if boundary is not None:
            if isinstance(boundary, sl.MultiLineString):
                for geom in boundary.geoms:
                    cv.polylines(
                        image,
                        [points_to_pixel(linestring_to_list(geom)).reshape(-1, 1, 2)[..., ::-1]],
                        False, (255, 0, 0), 2
                    )
            else:
                cv.polylines(
                    image,
                    [points_to_pixel(linestring_to_list(boundary)).reshape(-1, 1, 2)[..., ::-1]],
                    False, (255, 0, 0), 2
                )

        cv.imwrite(f"logs/navigation_{self.env_id}.png", image)
        path = pixel_to_points(path)
        path = path.tolist()
        return path
