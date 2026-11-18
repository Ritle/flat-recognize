"""Recover a connected wall network from a flat drawing with pale, outlined walls."""

import cv2
import numpy as np


def pale_wall_mask(rgb):
    gray = rgb.min(axis=2)
    neutral = np.ptp(rgb, axis=2) <= 2
    if np.mean(neutral & (gray >= 250)) < .55:
        return None
    histogram = np.bincount(gray[neutral].ravel(), minlength=256)
    level = 180 + int(histogram[180:246].argmax())
    if not .02 <= histogram[level] / gray.size <= .22:
        return None
    flat = (neutral & (gray == level)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(flat)
    mask = np.zeros_like(flat)
    backed = total = 0
    for i in range(1, count):
        x, y, w, h, area = map(int, stats[i])
        if area < 24 or max(w, h) < 12:
            continue
        # Check inside a padded crop so the ink on the edge of the band is visible.
        x0, y0 = max(0, x-1), max(0, y-1)
        cut = np.s_[y0:y+h+1, x0:x+w+1]
        part = (labels[cut] == i).astype(np.uint8)
        radius = cv2.distanceTransform(np.pad(part, 1), cv2.DIST_L2, 5).max()
        if radius > min(rgb.shape[:2]) * .035 or max(w, h) < 4 * radius:
            continue
        rim = cv2.dilate(part, np.ones((3, 3), np.uint8)) > part
        if rim.any() and np.mean(gray[cut][rim] < level - 12) >= .5:
            backed += area
        total += area
        mask[cut] |= part
    if total < histogram[level] * .85 or backed < total * .85:
        return None
    return cv2.dilate(mask, np.ones((3, 3), np.uint8)), level


def framed_bays(mask, gray, vertical):
    if vertical:
        mask, gray = mask.T, gray.T
    bands = {}
    for y, row in enumerate(mask):
        runs = np.flatnonzero(np.diff(np.r_[0, row, 0])).reshape(-1, 2)
        for (left, start), (end, right) in zip(runs, runs[1:]):
            if min(start-left, right-end) >= 2 and 12 <= end-start <= min(mask.shape) * .45:
                bands.setdefault((int(start-1), int(end)), []).append(y)
    result = []
    for (start, end), rows in bands.items():
        for band in np.split(rows, np.flatnonzero(np.diff(rows) != 1) + 1):
            if not 2 <= len(band) <= min((end-start) / 1.8, max(8, min(mask.shape) * .05)):
                continue
            top, bottom = int(band[0]), int(band[-1])
            strip = gray[:, start+2:end-1]
            def rail_at(y):
                return 0 <= y < len(gray) and np.mean(strip[y] < 244) >= .85
            if not (any(rail_at(y) for y in range(top-2, top+2))
                    and any(rail_at(y) for y in range(bottom-1, bottom+3))):
                continue
            # A crossbar can divide a door and fixed glazing in the same bay.
            # Require an additional longitudinal rail on only one side.
            spans = []
            current = start
            if bottom - top >= 5:
                interior = gray[top+1:bottom, start+2:end-1] < 240
                bars = np.flatnonzero(interior.mean(axis=0) >= .85) + start+2
                for group in np.split(bars, np.flatnonzero(np.diff(bars) != 1) + 1):
                    if not len(group) or len(group) > 4:
                        continue
                    at = round(float(group.mean()))
                    if min(at-current, end-at) < max(12, len(band)*2):
                        continue
                    middle = gray[top+2:bottom-1]
                    left_rail = (middle[:, current+3:at-2] < 240).mean(axis=1).max()
                    right_rail = (middle[:, at+3:end-2] < 240).mean(axis=1).max()
                    if min(left_rail, right_rail) < .2 and max(left_rail, right_rail) > .85:
                        spans.append((current, int(group[0]) - .5))
                        current = int(group[-1]) + .5
            for a, b in spans + [(current, end)]:
                x0, y0, x1, y1 = (top, a, bottom, b) if vertical else (a, top, b, bottom)
                core = gray[top+1:bottom, int(a)+2:int(b)-1]
                glazed = bool(core.size and np.max(np.mean(core < 235, axis=1)) >= .85)
                result.append(dict(outer=[[x0,y0],[x1,y0],[x1,y1],[x0,y1]], holes=[],
                                   source_class="outlined_wall_frame", glazing_rail=glazed))
    return result


def enclosed_areas(mask, minimum):
    n, _, stats, _ = cv2.connectedComponentsWithStats(np.pad((mask == 0).astype(np.uint8), 1,
                                                            constant_values=1))
    return [int(stats[i, 4]) for i in range(1, n) if stats[i, 0] and stats[i, 1] and stats[i, 4] >= minimum]


def outlined_wall_layout(rgb, predicted):
    palette = pale_wall_mask(rgb)
    if palette is None:
        return None
    mask, level = palette
    gray = rgb.min(axis=2)
    bays = framed_bays(mask, gray, False) + framed_bays(mask, gray, True)
    if len(bays) < 2:
        return None
    network = mask.copy()
    for bay in bays:
        cv2.fillPoly(network, [np.int32(bay["outer"])], 1)
    joined = cv2.morphologyEx(network, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    minimum = max(100, round(gray.size * .005))
    rooms = enclosed_areas(joined, minimum)
    # Only replace a leaky model mask when the ruled drawing supplies a single
    # connected network and independently encloses several substantial rooms.
    if (len(rooms) < 2 or sum(rooms) < gray.size * .25 or cv2.connectedComponents(joined)[0] != 2
            or sum(enclosed_areas(predicted, minimum)) >= sum(rooms) * .5):
        return None
    nearby = cv2.dilate(joined, np.ones((7, 7), np.uint8)) != 0
    predicted_structure = predicted != 0
    covered = cv2.dilate(predicted_structure.astype(np.uint8), np.ones((7, 7), np.uint8))
    if (covered[joined != 0].mean() < .25 or not predicted_structure.any()
            or nearby[predicted_structure].mean() < .85):
        return None
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    rings = [c.reshape(-1, 2).tolist() for c in contours]
    walls = [dict(outer=rings[i], holes=[rings[j] for j, node in enumerate(hierarchy[0]) if node[3] == i])
             for i, node in enumerate(hierarchy[0]) if node[3] == -1]
    return walls, bays, dict(fill_level=level, enclosed_regions=len(rooms))
