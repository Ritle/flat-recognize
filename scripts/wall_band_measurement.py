"""Measure structural wall bands beside openings and straight wall axes."""

import math
import statistics

from shapely.geometry import LineString, Point


def sample_straight_wall_band(walls, line, nominal_width):
    """Find one stable source-wall strip across five cuts of a straight axis."""
    if walls.is_empty or line.length < 30 or nominal_width <= 0:
        return None
    (ax, ay), (bx, by) = line.coords
    dx, dy = bx - ax, by - ay
    length = math.hypot(dx, dy)
    nx, ny = -dy / length, dx / length
    reach = max(100, nominal_width * 5)
    samples = []
    for fraction in (.15, .3, .5, .7, .85):
        x, y = ax + dx * fraction, ay + dy * fraction
        center = Point(x, y)
        cut = walls.intersection(LineString([
            (x - nx * reach, y - ny * reach),
            (x + nx * reach, y + ny * reach)]))
        pieces = ([cut] if cut.geom_type == "LineString" else
                  list(getattr(cut, "geoms", [])))
        nearby = [piece for piece in pieces
                  if piece.geom_type == "LineString" and 8 <= piece.length <= reach
                  and piece.distance(center) <= max(12, nominal_width * .75)]
        if not nearby:
            return None
        band = max(nearby, key=lambda piece: piece.length)
        middle = band.interpolate(.5, normalized=True)
        samples.append({"width": band.length,
                        "offset": (middle.x - x) * nx + (middle.y - y) * ny})
    widths = [sample["width"] for sample in samples]
    offsets = [sample["offset"] for sample in samples]
    if (max(widths) > min(widths) * 1.4
            or max(offsets) - min(offsets) > max(4, min(widths) * .2)):
        return None
    offset = statistics.median(offsets)
    if any(value * offset <= 0 for value in offsets):
        return None
    return {"median_width_px": statistics.median(widths),
            "median_offset_px": offset, "normal": (nx, ny)}


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
