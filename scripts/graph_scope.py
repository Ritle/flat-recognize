"""Validate explicitly excluded source-image regions for graph construction."""
import math

from shapely.geometry import Polygon


def validate_graph_scope(scope, image_size):
    if not isinstance(scope, dict) or set(scope) != {'exclude_polygons'}:
        raise ValueError('Scope must contain exclude_polygons')
    rings = scope['exclude_polygons']
    if not isinstance(rings, list) or len(rings) > 32:
        raise ValueError('Too many excluded regions')
    width, height = image_size
    for ring in rings:
        if not isinstance(ring, list) or not 3 <= len(ring) <= 64:
            raise ValueError('Invalid excluded polygon')
        for point in ring:
            if (not isinstance(point, list) or len(point) != 2
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in point)
                    or not 0 <= point[0] < width or not 0 <= point[1] < height):
                raise ValueError('Excluded polygon is outside the image')
        polygon = Polygon(ring)
        if not polygon.is_valid or polygon.area < 1:
            raise ValueError('Excluded polygon must be nonempty and valid')
    return rings
