"""Check window placement against closed physical room footprints.

Exterior means the building outline, not the image border or a bounding box.
Missing rooms and enclosed voids stay uncertain; this check never assigns a
replacement opening type or turns an unresolved opening into a solid wall.
"""
import math

from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union


def window_exterior_context(span, wall_width, rooms):
    shapes = [Polygon(r['polygon'], r.get('holes', [])) for r in rooms]
    shapes = [p for p in shapes if p.is_valid and not p.is_empty and p.area > 0]
    if not shapes:
        return dict(status='unknown', reason='no_closed_physical_rooms')
    footprint = unary_union(shapes)
    parts = [footprint] if footprint.geom_type == 'Polygon' else list(footprint.geoms)
    # Fill holes only to distinguish open exterior from an enclosed unknown void.
    envelope = unary_union([Polygon(p.exterior) for p in parts])
    boundary = unary_union([p.exterior for p in parts])
    line = LineString(span)
    if line.length <= 0 or not math.isfinite(wall_width) or wall_width <= 0:
        return dict(status='unknown', reason='invalid_opening_geometry')
    a, b = line.coords
    normal = (-(b[1]-a[1])/line.length, (b[0]-a[0])/line.length)
    sides = []
    for sign in (-1, 1):
        labels = []
        for fraction in (.25, .5, .75):
            center = line.interpolate(fraction, normalized=True)
            for offset in (wall_width/2+2, wall_width/2+4):
                p = Point(center.x+normal[0]*offset*sign, center.y+normal[1]*offset*sign)
                labels.append('inside' if footprint.contains(p) else 'outside' if not envelope.covers(p) else 'void')
        sides.append(labels[0] if len(set(labels)) == 1 else 'mixed')
    tolerance = max(2, min(3, wall_width*.25))
    boundary_fraction = line.intersection(boundary.buffer(tolerance)).length / line.length
    if sorted(sides) == ['inside', 'outside'] and boundary_fraction >= .9:
        status, reason = 'exterior', 'room_to_open_exterior'
    elif sides == ['inside', 'inside'] and boundary_fraction <= .1:
        status, reason = 'interior', 'physical_interior_on_both_sides'
    else:
        status, reason = 'unknown', 'incomplete_or_ambiguous_footprint'
    return dict(status=status, reason=reason, sides=sides,
                exterior_boundary_fraction=round(boundary_fraction, 4),
                rule='windows_normally_face_building_exterior', requires_type_review=status != 'exterior')
