"""Recover orthogonal glazing and swing doors drawn in a common CAD pen colour."""

from collections import defaultdict

import cv2
import numpy as np

from door_leaf_evidence import perpendicular_leaf_and_arc


def wall_fill(rgb):
    gray = rgb[:, :, 0]
    neutral = np.ptp(rgb, axis=2) <= 2
    if np.mean(neutral & (gray >= 250)) < .55:
        return None
    counts = np.bincount(gray[neutral], minlength=256)
    level = 95 + int(counts[95:210].argmax())
    if not .025 <= counts[level] / gray.size <= .22:
        return None
    exact = (neutral & (gray == level)).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(exact, cv2.MORPH_CLOSE, kernel)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(closed)
    keep = ((stats[:, cv2.CC_STAT_AREA] >= gray.size * .0005)
            & (np.minimum(stats[:, 2], stats[:, 3]) >= 3))
    keep[0] = False
    mask = keep[labels].astype(np.uint8)
    if mask.sum() < counts[level] * .85:
        return None
    return cv2.dilate(mask, kernel), level


def rectangle(start, end, low, high, vertical, **fields):
    x, y, xx, yy = (low, start, high, end) if vertical else (start, low, end, high)
    return dict(outer=[[x, y], [xx, y], [xx, yy], [x, yy]], holes=[], **fields)


def glazed_spans(ink, vertical):
    # Work along rows. Three independent parallel strokes distinguish a frame
    # from a leader, dimension line or the two outlines of a door leaf.
    field = ink.T if vertical else ink
    minimum = max(15, round(min(field.shape) * .035))
    lines = cv2.morphologyEx(field, cv2.MORPH_CLOSE, np.ones((1, 5), np.uint8))
    lines = cv2.morphologyEx(lines, cv2.MORPH_OPEN, np.ones((1, minimum | 1), np.uint8))
    grouped = cv2.morphologyEx(lines, cv2.MORPH_CLOSE, np.ones((7, 1), np.uint8))
    _, _, stats, _ = cv2.connectedComponentsWithStats(grouped)
    rails = [tuple(map(int, box[:4])) for box in stats[1:] if box[2] >= minimum]
    frames = []
    for x, y, w, h in rails:
        if h < 4 or w < h * 4:
            continue
        strokes = lines[y:y+h, x:x+w].mean(axis=1) >= .8
        if np.count_nonzero(np.diff(np.r_[False, strokes, False])) < 6:
            continue
        low, high = y, y+h-1
        # A parallel sill line with matching ends gives the opposite wall face.
        sills = [yy for xx, yy, ww, hh in rails if hh <= 2 and abs(xx-x) <= 3
                 and abs(xx+ww-x-w) <= 3 and h < abs(yy-y) <= min(w*.75, min(field.shape)*.08)]
        if len(sills) == 1:
            low, high = min(low, sills[0]), max(high, sills[0])
        frames.append(rectangle(x, x+w-1, low, high, vertical,
                                source_class="measured_glazing", glazing_rail=True))
    return frames


def swing_spans(network, ink, vertical):
    field = network.T if vertical else network
    pigment = ink.T if vertical else ink
    gaps = defaultdict(list)
    for y, row in enumerate(field):
        runs = np.flatnonzero(np.diff(np.r_[0, row, 0])).reshape(-1, 2)
        for (a, start), (end, b) in zip(runs, runs[1:]):
            if min(start-a, b-end) >= 5 and 20 <= end-start <= min(field.shape)*.25:
                gaps[(int(start-1), int(end))].append(y)
    swings = []
    for (start, end), rows in gaps.items():
        for band in np.split(rows, np.flatnonzero(np.diff(rows) != 1)+1):
            if not 4 <= len(band) <= min(end-start, min(field.shape)*.08):
                continue
            low, high = int(band[0]), int(band[-1])
            # Both jambs must carry this pen colour before sampling a swing.
            if any(not (pigment[low:high+1, max(0, at-4):at+5] == 0).any()
                   for at in (start, end)):
                continue
            # In a deep reveal the hinge can be drawn on either face.
            for axis in (low, high, (low+high)/2):
                a, b = ((axis, start), (axis, end)) if vertical else ((start, axis), (end, axis))
                swing = perpendicular_leaf_and_arc(ink, a, b, min(10, len(band)))
                if swing:
                    swings.append((low, high, start, end, swing))
                    break
    # Antialiasing on one jamb can move its end by a pixel halfway
    # through the wall. Combine adjacent bands only for the same leaf.
    merged = []
    for low, high, start, end, swing in sorted(swings, key=lambda s: s[:4]):
        for i, (y0, y1, a, b, other) in enumerate(merged):
            if (low == y1+1 and max(abs(start-a), abs(end-b)) <= 2
                    and np.linalg.norm(np.subtract(swing["tip"], other["tip"])) <= 4):
                best = max((swing, other), key=lambda e: (e["arc_hits"], e["leaf_support"]))
                merged[i] = (y0, high, min(start, a), max(end, b), best)
                break
        else:
            merged.append((low, high, start, end, swing))
    return [rectangle(a, b, low, high, vertical, source_class="measured_swing", raster_swing=swing)
            for low, high, a, b, swing in merged]


def glazed_corners(walls, horizontal, vertical):
    """Leave the common corner post solid where two glazed runs turn."""
    for along in horizontal:
        for across in vertical:
            (x0, y0), (x1, y1) = along["outer"][::2]
            (u0, v0), (u1, v1) = across["outer"][::2]
            if not (x0 <= u0 <= u1 <= x1 and v0 <= y0 <= y1 <= v1):
                continue
            if min(u0-x0, x1-u1) > 2 or min(y0-v0, v1-y1) > 2:
                continue
            walls[y0:y1+1, u0:u1+1] = 1
            lo, hi = (u1, x1) if u0-x0 <= 2 else (x0, u0)
            along["outer"] = rectangle(lo, hi, y0, y1, False)["outer"]
            lo, hi = (y1, v1) if y0-v0 <= 2 else (v0, y0)
            across["outer"] = rectangle(lo, hi, u0, u1, True)["outer"]


def measured_color_layout(rgb):
    hue, sat, _ = cv2.split(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV))
    saturated = sat > 80
    if saturated.mean() < .001:
        return None
    peak = int(np.bincount(hue[saturated], minlength=180).argmax())
    delta = np.abs(hue.astype(int) - peak)
    accent = ((sat > 60) & (np.minimum(delta, 180-delta) <= 4)).astype(np.uint8)
    if np.mean(accent[saturated]) < .5 or accent.mean() > .08:
        return None
    palette = wall_fill(rgb)
    if palette is None:
        return None
    walls, level = palette
    horizontal, vertical = glazed_spans(accent, False), glazed_spans(accent, True)
    glazed_corners(walls, horizontal, vertical)
    frames = horizontal + vertical
    if not frames:
        return None
    network = walls.copy()
    for frame in frames:
        cv2.fillPoly(network, [np.int32(frame["outer"])], 1)
    ink = (1-accent)*255
    doors = swing_spans(network, ink, False) + swing_spans(network, ink, True)
    if doors:
        return walls, frames + doors, level
    return None
