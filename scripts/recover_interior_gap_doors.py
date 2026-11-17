"""Resolve existing interior graph gaps using raster evidence at the measured jambs."""
import cv2
import numpy as np
from shapely.geometry import LineString, Point

from classify_openings import structural_union
from door_leaf_evidence import dark_stroke_support
from mask_wall_graph import clear_jamb_span
from reclassify_graph_doors import jamb_face_leaves, raster_aperture
from review_graph_doors import gap_fingerprint, insert_supported_internal_doors
from window_frame_evidence import narrow_frame_rails


def interior_gap_leaf(gray, distance, gap, walls, scale, other_openings):
    width = gap['local_width_px']
    model = clear_jamb_span(LineString(gap['span']), walls, width, scale)
    if model is None:
        return None
    measured = raster_aperture(distance, *model.coords, width)
    if measured is None or model.hausdorff_distance(LineString(measured)) > max(2, width*.35):
        return None
    if narrow_frame_rails(gray, *measured, width):
        return None
    measured = np.asarray(measured)
    axis = measured[1]-measured[0]
    length = np.linalg.norm(axis)
    axis /= length
    normal = np.array([-axis[1], axis[0]])
    options = []
    for corner, leaf in jamb_face_leaves(gray, measured, normal, width):
        tip = np.asarray(leaf['tip'])
        if (leaf['jamb_distance_px'] > max(2, width/2)
                or not 0 < (tip-measured[0]) @ axis < length
                or walls.distance(Point(tip)) < width*.35
                or dark_stroke_support(gray, corner, tip) < .95):
            continue
        # Do not borrow a stroke rooted on another known opening, even if it
        # falls inside the same search patch as the proposed jamb.
        if any(line.distance(Point(leaf['hinge'])) <= max(2, leaf['jamb_distance_px'])
               for line in other_openings):
            continue
        options.append(dict(leaf, jamb_corner=corner.tolist()))
    if options:
        return dict(leaf=min(options, key=lambda item: item['jamb_distance_px']),
                    measured_jamb_span=measured.tolist(), model_clear_span=list(model.coords))


def recover_interior_gap_doors(graph, source, rgb):
    gaps = [g for g in graph['diagnostics'].get('topology_gaps', [])
            if g.get('reason') == 'interior_raster_line_without_confirmed_type']
    if not gaps:
        return graph, []
    if (graph.get('coordinate_space') != 'source_raster'
            or source['meta'].get('coordinate_space','source_raster') != 'source_raster'
            or (source['meta']['width'],source['meta']['height']) != rgb.shape[1::-1]):
        raise ValueError('Gap recovery requires matching source-raster coordinates')
    walls = structural_union(source.get('walls', []))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    distance = cv2.distanceTransform((rgb.max(axis=2)<160).astype(np.uint8), cv2.DIST_L2, 5)
    openings = [LineString(b.get('clear_span', b['span'])) for b in graph['diagnostics'].get('opening_bridges', [])]
    boundaries = [(g,LineString(g['span'])) for g in graph['diagnostics']['topology_gaps']]
    result, rows = graph, []
    for gap in gaps:
        row = dict(span=gap['span'], accepted=False)
        rows.append(row)
        neighbours = openings + [line for other,line in boundaries if other is not gap]
        evidence = interior_gap_leaf(gray, distance, gap, walls, graph['scale_cm_per_px'], neighbours)
        if evidence is None:
            row['reason'] = 'missing_independent_jamb_leaf'
            continue
        fingerprint = gap_fingerprint(gap)
        evidence.update(gap_fingerprint=fingerprint, local_width_px=gap['local_width_px'])
        metadata = dict(opening_id='gap-door-'+fingerprint, source='interior_jamb_face_leaf', evidence=evidence)
        try:
            result = insert_supported_internal_doors(result, source, rgb, [(gap, metadata)])
        except ValueError as exc:
            row['reason'] = str(exc)
            continue
        row.update(accepted=True, evidence=evidence)
    return result, rows
