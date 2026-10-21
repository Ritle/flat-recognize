"""Measure structural wall bands adjacent to a bounded opening."""

import statistics

from shapely.geometry import LineString, Point


def flanking_wall_band(walls, span):
    """Measure wall-mask thickness and axis just beyond both ends of a bay."""
    if walls.is_empty:
        return None
    (x0, y0), (x1, y1) = span
    horizontal = abs(x1 - x0) >= abs(y1 - y0)
    lo, hi = sorted((x0, x1) if horizontal else (y0, y1))
    center = (y0 + y1) / 2 if horizontal else (x0 + x1) / 2
    length = hi - lo
    reach = max(100, length * .6)
    offsets = (5, 10, 20)
    sides, axes = {}, {}
    for name, sign, endpoint in (("before", -1, lo), ("after", 1, hi)):
        widths, centers = [], []
        for offset in offsets:
            position = endpoint + sign * offset
            line = (LineString([(position, center - reach),
                                (position, center + reach)]) if horizontal else
                    LineString([(center - reach, position),
                                (center + reach, position)]))
            sample = walls.intersection(line)
            pieces = ([sample] if sample.geom_type == "LineString" else
                      list(getattr(sample, "geoms", [])))
            nearby = [piece for piece in pieces
                      if piece.geom_type == "LineString" and piece.length >= 5
                      and piece.distance(Point(position, center) if horizontal else
                                         Point(center, position)) <= 15]
            if nearby:
                best = max(nearby, key=lambda piece: piece.length)
                widths.append(best.length)
                centers.append(best.centroid.y if horizontal else best.centroid.x)
        if widths:
            sides[name] = round(statistics.median(widths), 2)
            axes[name] = round(statistics.median(centers), 2)
    if not sides:
        return None
    consistent = (len(sides) == 2 and
                  max(sides.values()) <= min(sides.values()) * 1.5)
    return {"sampled_sides_px": sides,
            "sampled_axes_px": axes,
            "median_thickness_px": round(statistics.median(sides.values()), 2),
            "consistent_flanks": consistent}
