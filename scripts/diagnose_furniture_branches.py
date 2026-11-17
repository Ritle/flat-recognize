"""Measure short dangling wall branches on the source raster for visual review."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import polygonize, unary_union


def terminal_paths(edges):
    neighbours = defaultdict(list)
    for edge in edges:
        a, b = edge['start'], edge['end']
        neighbours[a].append(b)
        neighbours[b].append(a)
    seen = set()
    for tip, adjacent in neighbours.items():
        if len(adjacent) != 1 or tip in seen:
            continue
        path = [tip, adjacent[0]]
        while len(neighbours[path[-1]]) == 2:
            left, right = neighbours[path[-1]]
            path.append(right if left == path[-2] else left)
        isolated = len(neighbours[path[-1]]) == 1
        if isolated:
            seen.add(path[-1])
        yield path, isolated


def stroke_evidence(line, exposed, width, ink, labels, stats):
    distances = np.linspace(0, exposed, max(2, int(np.ceil(exposed)) + 1))
    radii = []
    nearby_radii = []
    ink_hits = 0
    tolerance = max(1, min(2, round(width / 4)))
    nearby_labels = set()
    margin = max(1, round(width / 2))
    for distance in distances:
        sample = line.interpolate(distance)
        x, y = round(sample.x), round(sample.y)
        if 0 <= y < ink.shape[0] and 0 <= x < ink.shape[1]:
            near_axis = ink[max(0, y-tolerance):y+tolerance+1, max(0, x-tolerance):x+tolerance+1]
            ink_hits += bool(near_axis.max())
            radii.append(float(ink[y, x]))
            patch = np.s_[max(0, y-margin):y+margin+1, max(0, x-margin):x+margin+1]
            nearby_labels.update(np.unique(labels[patch]))
            nearby_radii.append(float(ink[patch].max()))
    core = sum(r >= max(1.5, .30 * width) for r in radii) / len(radii) if radii else None
    # A skeleton spur can run down the margin of a filled wall, particularly
    # at a corner. Its centreline alone then resembles a thin stroke.
    mass = (sum(r >= max(2., .40 * width) for r in nearby_radii) / len(nearby_radii)
            if nearby_radii else None)
    # A small enclosed light component can belong to a fixture, a cabinet
    # or a printed glyph. Record this as visual context for review.
    loops = []
    for label in sorted(nearby_labels - {0}):
        x, y, w, h, area = map(int, stats[label])
        if (x > 0 and y > 0 and x+w < ink.shape[1] and y+h < ink.shape[0]
                and min(w, h) >= 2 and max(w, h) <= 4*width
                and .15*width**2 <= area and area >= .25*w*h):
            loops.append([x, y, w, h])
    return dict(thick_ink_fraction=core, nearby_solid_fraction=mass,
                raster_support_fraction=ink_hits / len(distances),
                samples_in_bounds=len(radii) == len(distances), nearby_closed_details=loops)


def measure_branches(graph, rgb):
    if graph['coordinate_space'] != 'source_raster':
        raise ValueError('Source raster coordinates required')
    nodes = {n['id']: n['point'] for n in graph['nodes']}
    edges = graph['edges']
    local_widths = defaultdict(list)
    lines = []
    for e in edges:
        a, b = e['start'], e['end']
        local_widths[a].append((b, e['thickness_px']))
        local_widths[b].append((a, e['thickness_px']))
        lines.append(LineString([nodes[a], nodes[b]]))
    rooms = list(polygonize(lines))
    typical = np.median([e['thickness_px'] for e in edges]) if edges else 0
    diag = graph['diagnostics']
    protected = [LineString(r['span']) for r in
                 diag.get('opening_bridges', []) + diag.get('topology_gaps', [])]
    protected.extend(Polygon(p) for p in diag.get('excluded_regions', []))
    protected = unary_union(protected)
    doors = [LineString(b.get('clear_span', b['span'])) for b in diag.get('opening_bridges', [])
             if b['kind'] in ('doors', 'recovered_door')]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    ink = cv2.distanceTransform((gray < 150).astype('uint8'), cv2.DIST_L2, 5)
    _, labels, stats, _ = cv2.connectedComponentsWithStats((gray >= 150).astype('uint8'), connectivity=4)
    rows = []
    for ids, isolated in terminal_paths(edges):
        path = [nodes[i] for i in ids]
        line = LineString(path)
        adjacent = [w for n, w in local_widths[ids[-1]] if n != ids[-2]]
        width = float(np.median(adjacent)) if adjacent else float(typical)
        if line.length == 0 or width <= 0 or line.length > 5 * width:
            continue
        # Keep the junction's solid ink out of the measured stroke.
        exposed = line.length if isolated else max(0., line.length - width / 2)
        evidence = stroke_evidence(line, exposed, width, ink, labels, stats)
        in_swing = any(min(line.distance(Point(p)) for p in door.coords) <= door.length
                       for door in doors)
        inside = any(room.contains(line.interpolate(.5, normalized=True)) and room.covers(line)
                     for room in rooms)
        clearance = None if protected.is_empty else line.distance(protected)
        if clearance is not None and clearance <= width:
            reason = 'near_opening_gap_or_exclusion'
        elif in_swing:
            reason = 'within_possible_door_swing'
        elif not inside:
            reason = 'outside_or_incomplete_room'
        elif exposed < 2 or not evidence['samples_in_bounds']:
            reason = 'insufficient_exposed_stroke'
        elif evidence['raster_support_fraction'] < .65:
            reason = 'missing_raster_stroke'
        elif evidence['thick_ink_fraction'] > .25:
            reason = 'substantial_wall_ink'
        elif evidence['nearby_solid_fraction'] >= .65:
            reason = 'alongside_solid_ink'
        elif evidence['nearby_closed_details']:
            reason = 'thin_branch_near_closed_detail'
        else:
            reason = 'thin_interior_branch'
        rows.append(dict(path=path, isolated=isolated, length_px=line.length,
                         reference_width_px=width, exposed_px=exposed, **evidence,
                         protected_distance_px=clearance, inside_physical_room=inside,
                         within_possible_door_swing=in_swing, reason=reason,
                         review_candidate=reason == 'thin_branch_near_closed_detail'))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path)
    parser.add_argument('graph', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--overlay', type=Path, help='PNG with numbered review candidates on the source raster')
    args = parser.parse_args()
    outputs = [p.resolve() for p in (args.output, args.overlay) if p is not None]
    if len(set(outputs)) != len(outputs) or set(outputs) & {args.image.resolve(), args.graph.resolve()}:
        parser.error('Expected distinct input and output paths')
    if args.overlay and args.overlay.suffix.lower() != '.png':
        parser.error('Expected a PNG overlay')
    graph = json.loads(args.graph.read_text(encoding='utf-8'))
    with Image.open(args.image) as im:
        raster = im.convert('RGB')
    rows = measure_branches(graph, np.asarray(raster))
    if args.overlay:
        draw = ImageDraw.Draw(raster)
        for index, row in enumerate(rows, 1):
            if row['review_candidate']:
                points = [tuple(point) for point in row['path']]
                draw.line(points, fill='#d01465', width=3)
                x, y = points[0]
                draw.text((x+3, y-9), str(index), fill='white', stroke_width=1, stroke_fill='#681137')
        args.overlay.parent.mkdir(parents=True, exist_ok=True)
        raster.save(args.overlay)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
