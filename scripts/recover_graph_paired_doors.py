"""Recover an interior paired door when both leaf roots became dangling wall edges."""
from collections import Counter
from itertools import combinations

import cv2
import numpy as np
from shapely.geometry import LineString, Point, Polygon

from reclassify_graph_doors import connected_leaf_ink, door_evidence, free_leaf_tip
from review_graph_door_branches import remove_jamb_branches
from review_graph_doors import gap_fingerprint, insert_supported_internal_doors


def branch_pairs(graph):
    points = {n['id']: np.asarray(n['point'], float) for n in graph['nodes']}
    neighbours = {uid: [] for uid in points}
    for edge in graph['edges']:
        a, b = edge['start'], edge['end']
        neighbours[a].append((b, edge['thickness_px']))
        neighbours[b].append((a, edge['thickness_px']))
    jambs = []
    for terminal, links in neighbours.items():
        if len(links) != 1:
            continue
        root, _ = links[0]
        if len(neighbours[root]) != 2:
            continue
        wall, width = next(link for link in neighbours[root] if link[0] != terminal)
        branch = points[terminal] - points[root]
        carrier = points[root] - points[wall]
        if 0 < np.linalg.norm(branch) <= 1.5*width and np.linalg.norm(carrier) >= width:
            jambs.append((points[root], points[terminal], carrier/np.linalg.norm(carrier), width))
    for first, second in combinations(sorted(jambs, key=lambda jamb: tuple(jamb[0])), 2):
        a, b = first[0], second[0]
        length = np.linalg.norm(b-a)
        width = min(first[3], second[3])
        if not 2*width < length <= 12*width:
            continue
        axis = (b-a)/length
        if first[2] @ axis < .98 or second[2] @ -axis < .98:
            continue
        yield dict(span=[a.tolist(), b.tolist()], local_width_px=width), [
            [root.tolist(), tip.tolist()] for root, tip, _, _ in (first, second)]


def branch_matches_leaf(branch, leaf, gray, width):
    root, end = np.asarray(branch)
    hinge, tip = np.asarray(leaf['hinge']), np.asarray(leaf['tip'])
    stroke = tip-hinge
    stub = end-root
    return (stub @ stroke >= .8 * np.linalg.norm(stub) * np.linalg.norm(stroke)
            and Point(end).distance(LineString([hinge,tip])) <= max(2, width/2)
            and (tip-end) @ stroke >= width*np.linalg.norm(stroke)
            and connected_leaf_ink(gray, root, hinge) and connected_leaf_ink(gray, hinge, tip)
            and free_leaf_tip(gray, hinge, tip, width))


def recover_graph_paired_doors(graph, source, rgb):
    if (source.get('meta', {}).get('width'), source.get('meta', {}).get('height')) != rgb.shape[1::-1]:
        raise ValueError('Source dimensions do not match raster')
    if graph.get('coordinate_space') != 'source_raster' or source['meta'].get('coordinate_space','source_raster') != 'source_raster':
        raise ValueError('Paired door recovery requires source_raster coordinates')
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    distance = cv2.distanceTransform((rgb.max(axis=2)<160).astype(np.uint8), cv2.DIST_L2, 5)
    rows, proposals = [], []
    protected = [LineString(g['span']) for g in graph['diagnostics'].get('topology_gaps', [])]
    protected.extend(Polygon(p) for p in graph['diagnostics'].get('excluded_regions', []))
    carriers = [LineString(b['span']) for b in graph['diagnostics'].get('opening_bridges', [])]
    for gap, branches in branch_pairs(graph):
        width, span = gap['local_width_px'], gap['span']
        row = dict(span=span, branches=branches, accepted=False)
        rows.append(row)
        evidence = door_evidence(gray, distance, span, width, paired_leaf_filter=lambda index, leaf:
                                 branch_matches_leaf(branches[index], leaf, gray, width))
        if not evidence:
            row['reason'] = 'missing_two_branch_aligned_free_leaves'
            continue
        if any(LineString(s).intersects(area) for s in [span, *branches] for area in protected):
            row['reason'] = 'intersects_exclusion_or_unresolved_boundary'
            continue
        if any(LineString(s).intersection(carrier).length > 0 for s in [span, *branches] for carrier in carriers):
            row['reason'] = 'overlaps_existing_opening_carrier'
            continue
        proposals.append((gap, branches, evidence, row))
    used = Counter(tuple(p) for gap, _, _, _ in proposals for p in gap['span'])
    result = graph
    for gap, branches, evidence, row in proposals:
        if any(used[tuple(p)] != 1 for p in gap['span']):
            row['reason'] = 'ambiguous_jamb_pair'
            continue
        fingerprint = gap_fingerprint(gap)
        metadata = dict(opening_id='paired-door-'+fingerprint, source='paired_raster_leaf_branches',
                        type_review=dict(evidence=evidence), evidence=dict(
                            gap_fingerprint=fingerprint, removed_dangling_branches=branches))
        try:
            trial = remove_jamb_branches(result, gap['span'], gap['local_width_px'], branches)
            trial = insert_supported_internal_doors(trial, source, rgb, [(gap, metadata)])
        except ValueError as exc:
            row['reason'] = str(exc)
            continue
        clear = LineString(trial['diagnostics']['opening_bridges'][-1]['clear_span'])
        measured = LineString(evidence['measured_jamb_span'])
        if clear.hausdorff_distance(measured) > max(2, gap['local_width_px']/2):
            row['reason'] = 'raster_and_model_jambs_disagree'
            continue
        result = trial
        row.update(accepted=True, evidence=evidence)
    return result, rows
