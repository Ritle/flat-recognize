"""Fit orthogonal room rings to wall ink and reviewed openings in source pixels."""

from collections import Counter

import cv2
import numpy as np
from shapely.geometry import LineString, Polygon

from extract_rooms_v2 import wall_mask


def measured_axis(field, near, lo, hi, direction, reach, carriers, minimum_fraction=None):
    """Vote with wall-band midpoints along the outside of a room edge."""
    votes = Counter()
    for along in range(max(0, lo), min(hi + 1, field.shape[1])):
        scan = field[:, along]
        offset = max(0, near - reach * 2)
        stop = min(len(scan), near + reach * 2 + 1)
        bands = np.flatnonzero(np.diff(np.r_[False, scan[offset:stop], False])).reshape(-1, 2)
        candidates = []
        for left, right in bands:
            center = offset + (left + right - 1) / 2
            distance = (center - near) * direction
            if right - left >= 2 and -1 <= distance <= reach and left > 0 and offset + right < stop:
                candidates.append((abs(distance), center))
        if candidates:
            votes[min(candidates)[1]] += 1
    if votes:
        axis, count = votes.most_common(1)[0]
        supported = any(abs(center - axis) <= 2 for center in carriers)
        fraction = minimum_fraction if minimum_fraction is not None else (.2 if supported else .4)
        if count >= max(3, (hi - lo + 1) * fraction):
            return axis
    return None


def confirmed_interior_gaps(source):
    return [o for key in ("doors", "passages") for o in source.get(key, [])
            if o.get("source_class") == "unmarked_interior_gap"
            and o.get("classification", {}).get("user_confirmed")]


def band_axes(field, reach):
    """Rank wall axes by the number of cross-sections that bisect a thin band."""
    votes = Counter()
    for row in field.T:
        bands = np.flatnonzero(np.diff(np.r_[0, row, 0])).reshape(-1, 2)
        for a, b in bands:
            if 2 <= b-a <= reach*2:
                votes[(int(a+b)-1) / 2] += 1
    return [axis for axis, n in votes.most_common() if n >= 4]


def reviewed_room_axes(source, shapes, gray, half_wall):
    interior = confirmed_interior_gaps(source)
    outlined = source.get("meta", {}).get("outlined_wall_layout")
    if (not interior and not outlined) or gray is None:
        return None
    if any(not p.is_valid or p.area <= 0 or p.interiors for p in shapes):
        return None
    walls = wall_mask(source, *gray.shape)
    ink = (gray < 140) & (cv2.dilate(walls, np.ones((3, 3), np.uint8)) > 0)
    # In gaps the symbol or review supplies the wall axis directly. Keep
    # those votes independent of the line thickness drawn for a symbol.
    openings = []
    for key in ("doors", "passages", "windows"):
        for item in source.get(key, []):
            x1, y1, x2, y2 = Polygon(item["outer"]).bounds
            if x2 - x1 >= y2 - y1:
                openings.append((True, (y1 + y2) / 2, x1, x2))
            else:
                openings.append((False, (x1 + x2) / 2, y1, y2))
    fitted = []
    reach = max(8, round(half_wall * 7))
    if outlined:
        # All these frames have measured jambs and continuous rails. Vote across
        # the complete network so adjoining rooms pick the same carrier at a T.
        network = walls.copy()
        for item in (source.get("doors", []) + source.get("windows", []) + source.get("passages", [])):
            cv2.fillPoly(network, [np.int32(item["outer"])], 255)
        ink = cv2.morphologyEx(network, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
        reach = max(8, round(min(gray.shape) * .07))
        directions = {True: band_axes(ink, reach), False: band_axes(ink.T, reach)}
    for shape in shapes:
        ring = np.asarray(shape.exterior.coords[:-1], dtype=np.float32)
        epsilon = max(2, half_wall * 2)
        if outlined:
            epsilon = max(2, min(gray.shape) * .005) if outlined.get("colored_symbols") else 2
        ring = cv2.approxPolyDP(ring, epsilon, True).reshape(-1, 2)
        if len(ring) < 4:
            return None
        axes = []
        ccw = Polygon(ring).exterior.is_ccw
        for a, b in zip(ring, np.roll(ring, -1, axis=0)):
            dx, dy = b - a
            horizontal = abs(dx) > abs(dy)
            if min(abs(dx), abs(dy)) > max(abs(dx), abs(dy)) * .2:
                return None
            near = round(float((a[1] + b[1] if horizontal else a[0] + b[0]) / 2))
            along = 0 if horizontal else 1
            lo, hi = sorted((round(float(a[along])), round(float(b[along]))))
            direction = np.sign(-dx if horizontal else dy) * (1 if ccw else -1)
            carriers = [center for orient, center, start, end in openings if orient == horizontal
                        and -1 <= (center - near) * direction <= reach
                        and min(hi, end) - max(lo, start) >= (hi - lo) * .5]
            field = ink if horizontal else ink.T
            if outlined:
                axis = None
                if outlined.get("colored_symbols"):
                    axis = measured_axis(field, near, lo+1, hi-1, direction, reach, carriers,
                                         minimum_fraction=.18)
                if axis is None:
                    axis = next((pos for pos in directions[horizontal]
                                 if -1 <= (pos-near)*direction <= reach and hi > lo + 2
                                 and field[round(pos), lo+1:hi].mean() >= .78), None)
                if axis is None and outlined.get("colored_symbols") and 5 < hi-lo <= reach:
                    # A short step along the face of a solid pier can have no
                    # transverse centreline. The adjacent long sides set its ends.
                    axis = next((near + step * int(direction) for step in range(4)
                                 if 0 <= near + step * int(direction) < len(field)
                                 and field[near + step * int(direction), lo+2:hi-1].all()), None)
            else:
                axis = measured_axis(field, near, lo + 1, hi - 1, direction, reach, carriers)
            if axis is None:
                if len(carriers) != 1:
                    return None
                axis = carriers[0]
            if axes and axes[-1][0] == horizontal:
                return None
            axes.append((horizontal, axis))
        if axes[0][0] == axes[-1][0]:
            return None
        corners = [(previous[1], axis) if horizontal else (axis, previous[1])
                   for previous, (horizontal, axis) in zip(axes[-1:] + axes[:-1], axes)]
        room = Polygon(corners)
        if outlined and outlined.get("colored_symbols"):
            room = room.simplify(0)
        if not room.is_valid or room.is_empty or not room.covers(shape.representative_point()):
            return None
        if shape.difference(room).area > shape.area * .02 or shape.boundary.hausdorff_distance(room.boundary) > reach * 1.5:
            return None
        if any(room.intersection(other).area > 1e-5 for other in fitted):
            return None
        fitted.append(room)
    # Each confirmed internal opening must separate two faces on the same axis.
    for door in interior:
        x1, y1, x2, y2 = Polygon(door["outer"]).bounds
        x, y = (x1 + x2) / 2, (y1 + y2) / 2
        line = (LineString([(x1 + 1, y), (x2 - 1, y)]) if x2 - x1 > y2 - y1
                else LineString([(x, y1 + 1), (x, y2 - 1)]))
        owners = [room for room in fitted if room.boundary.buffer(1.5).covers(line)]
        if len(owners) != 2 or not owners[0].boundary.intersection(owners[1].boundary).buffer(1.5).covers(line):
            return None
    return fitted
