"""Measure dark wall bands and ruled windows around surveyed opening symbols."""

from collections import Counter, defaultdict

import cv2
import numpy as np
from shapely.geometry import box
from shapely.ops import unary_union

from measured_color_layout import rectangle


def dark_wall_bands(gray, predicted, boxes):
    # Preserve narrow wall stubs at the ends of reviewed bays, even if the
    # network missed them. Detached cars, lettering and terraces lack support.
    kernel = np.ones((3, 3), np.uint8)
    ink = cv2.morphologyEx((gray < 56).astype(np.uint8), cv2.MORPH_OPEN, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink)
    support = cv2.dilate(predicted.astype(np.uint8), np.ones((9, 9), np.uint8))
    for x, y, xx, yy in boxes:
        support[max(0,y-3):yy+4, max(0,x-3):xx+4] = 1
    supported = np.bincount(labels.ravel(), weights=support.ravel(), minlength=count)
    keep = (stats[:, 4] >= 24) & (supported >= stats[:, 4] * .45)
    keep[0] = False
    return keep[labels].astype(np.uint8)


def ruled_windows(walls, gray, vertical, max_length=None):
    field, raster = (walls.T, gray.T) if vertical else (walls, gray)
    if max_length is None:
        max_length = min(walls.shape)*.25
    gaps = defaultdict(list)
    for y, row in enumerate(field):
        runs = np.flatnonzero(np.diff(np.r_[0, row, 0])).reshape(-1, 2)
        for (left, start), (end, right) in zip(runs, runs[1:]):
            if min(start-left, right-end) >= 3 and 12 <= end-start <= max_length:
                gaps[(int(start-1), int(end))].append(y)
    windows = []
    for (start, end), rows in gaps.items():
        for band in np.split(rows, np.flatnonzero(np.diff(rows) != 1)+1):
            lo, hi = int(band[0]), int(band[-1])
            if not 5 <= hi-lo <= (end-start)*.8:
                continue
            # A closed glazing symbol has both outer rails and at least two
            # separate inner strokes. Dimension lines do not fill all four.
            strip = raster[max(0,lo-1):hi+2, start+2:end-1]
            rails = (strip < 225).mean(axis=1) >= .83
            if (rails[:3].any() and rails[-3:].any()
                    and np.count_nonzero(np.diff(np.r_[False, rails, False])) >= 8):
                windows.append(rectangle(start, end, lo, hi, vertical,
                                         source_class="measured_glazing", glazing_rail=True))
    return windows


def measured_frame_at(gray, outer):
    """Check the model's proposed bay itself for continuous glazing between solid jambs."""
    coords = np.asarray(outer)
    x, y = np.floor(coords.min(axis=0)).astype(int)
    xx, yy = np.ceil(coords.max(axis=0)).astype(int)
    length = max(xx-x, yy-y)
    if length < 16 or min(xx-x, yy-y) < 4:
        return None
    x0, y0 = max(0, x-12), max(0, y-12)
    region = gray[y0:yy+13, x0:xx+13]
    ink = cv2.morphologyEx((region < 56).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    for frame in ruled_windows(ink, region, yy-y > xx-x, max_length=length+8):
        (a, b), (c, d) = frame["outer"][::2]
        expected = np.array([x-x0, y-y0, xx-x0, yy-y0])
        if np.max(np.abs(expected - [a, b, c, d])) <= 4:
            return {"bbox": [int(a+x0), int(b+y0), int(c+x0), int(d+y0)], "method": "four_glazing_rails"}
    return None


def reviewed_ink_layout(rgb, predicted, review):
    gray = rgb.min(axis=2)
    openings = review["openings"]
    boxes = [item["box"] for item in openings]
    empty = review.get("empty_regions", [])
    height, width = gray.shape
    for x, y, xx, yy in boxes + empty:
        if not (0 <= x < xx < width and 0 <= y < yy < height):
            raise ValueError("Reviewed rectangle is outside the source image")
    walls = dark_wall_bands(gray, predicted, boxes)
    for x, y, xx, yy in empty:
        walls[y:yy+1, x:xx+1] = 0
    distance = cv2.distanceTransform((walls == 0).astype(np.uint8), cv2.DIST_L2, 5)
    items = []
    for entry in openings:
        x, y, xx, yy = entry["box"]
        mx, my = round((x+xx)/2), round((y+yy)/2)
        jambs = (distance[my,x], distance[my,xx]) if xx-x >= yy-y else (distance[y,mx], distance[yy,mx])
        if max(jambs) > 3:
            raise ValueError("Reviewed opening has no wall at one of its jambs")
        item = rectangle(x, xx, y, yy, False, source_class="outlined_wall_frame")
        if entry.get("swing"):
            item["source_class"] = "surveyed_door"
        if "clear_span" in entry:
            lo, hi = entry["clear_span"]
            start, end = (x, xx) if xx-x >= yy-y else (y, yy)
            if not start <= lo < hi <= end:
                raise ValueError("Reviewed clear span is outside its opening")
            item["clear_span"] = [lo, hi]
        items.append(item)
    frames = ruled_windows(walls, gray, False) + ruled_windows(walls, gray, True)
    for frame in frames:
        (x, y), (xx, yy) = frame["outer"][::2]
        if not any(min(xx, b)-max(x, a) > 0 and min(yy, d)-max(y, c) > 0
                   for a, c, b, d in boxes + empty):
            items.append(frame)
    contours, hierarchy = cv2.findContours(walls, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        raise ValueError("Reviewed symbols have no supporting raster walls")
    rings = [c.reshape(-1, 2).tolist() for c in contours]
    polygons = [dict(outer=rings[i], holes=[rings[j] for j, child in enumerate(hierarchy[0])
                                           if child[3] == i])
                for i, node in enumerate(hierarchy[0]) if node[3] == -1]
    return polygons, items, {"reviewed_ink": True}


def survey_axes(field, carriers):
    votes = Counter()
    for row in field.T:
        for low, high in np.flatnonzero(np.diff(np.r_[0, row, 0])).reshape(-1, 2):
            if 2 <= high-low <= min(field.shape)*.05:
                votes[(low+high-1)/2] += 1
    axes = []
    # One axis serves both rooms even if a pocket or frame has two dark rails.
    for pos, depth, length in sorted(carriers, key=lambda c: -c[2]):
        if all(abs(pos-axis) > max(3, depth/2) for axis in axes):
            axes.append(pos)
    for pos, count in votes.most_common():
        if count < 10:
            break
        if (all(abs(pos-axis) > 4 for axis in axes)
                and all(abs(pos-center) > depth/2+1 for center, depth, _ in carriers)):
            axes.append(pos)
    return sorted({0, field.shape[0]-1, *axes})


def surveyed_room_faces(walls, items, shapes):
    """Assign cells between evidenced wall axes to the enclosing raster component."""
    from extract_rooms_v2 import known_opening_barrier

    network = walls.copy()
    carriers = [[], []]
    for item in items:
        coords = np.int32(item["outer"])
        cv2.fillPoly(network, [coords], 255)
        low, high = coords.min(axis=0), coords.max(axis=0)
        length_axis = int(np.argmax(high-low))
        across = 1-length_axis
        carriers[across].append(((low[across]+high[across])/2, high[across]-low[across],
                                 high[length_axis]-low[length_axis]))
    xx = survey_axes(network.T, carriers[0])
    yy = survey_axes(network, carriers[1])
    free = known_opening_barrier(walls, items) == 0
    count, labels = cv2.connectedComponents(free.astype(np.uint8))
    cells = defaultdict(list)
    for x, right in zip(xx, xx[1:]):
        for y, bottom in zip(yy, yy[1:]):
            region = labels[round(y):round(bottom), round(x):round(right)]
            votes = np.bincount(region.ravel(), minlength=count)
            votes[0] = 0
            if votes.any():
                cells[votes.argmax()].append(box(x, y, right, bottom))
    fitted = []
    owners = set()
    for source in shapes:
        point = source.representative_point()
        label = labels[round(point.y), round(point.x)]
        if not label or label in owners:
            return None
        face = unary_union(cells[label]).simplify(0)
        if (face.geom_type != "Polygon" or face.interiors or not face.is_valid
                or source.difference(face).area > source.area*.025
                or not face.covers(point)):
            return None
        owners.add(label)
        fitted.append(face)
    return fitted
