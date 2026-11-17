"""Locate a door between two short parallel-lined panels for graph review."""
import cv2
import numpy as np
from shapely import STRtree
from shapely.geometry import LineString, Point, box

from reclassify_graph_doors import jamb_face_leaves, raster_aperture
from window_frame_evidence import narrow_frame_rails


def wall_anchor(jamb, direction, reach, lines, tree):
    ray = LineString([jamb, jamb + direction*reach])
    hits = []
    for index in tree.query(ray, predicate='intersects'):
        hit = ray.intersection(lines[index])
        hits.append((Point(jamb).distance(hit), int(index), hit))
    if not hits:
        return None
    _, index, hit = min(hits, key=lambda h: h[0])
    if hit.geom_type != 'Point':
        return None
    a,b = lines[index].coords
    if abs((np.asarray(b)-a) @ direction) > .35 * lines[index].length:
        return None
    return np.asarray(hit.coords[0]), index


def raster_partitions(graph, rgb):
    """Yield measured geometry only; parallel strokes do not determine wall material."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    distance = cv2.distanceTransform((gray < 150).astype(np.uint8), cv2.DIST_L2, 5)
    _, _, stats, _ = cv2.connectedComponentsWithStats((distance >= 2).astype(np.uint8))
    posts = []
    for x,y,w,h,area in stats[1:]:
        if min(w,h) >= 3 and max(w,h) <= 1.8*min(w,h) and area >= .65*w*h:
            posts.append((np.array([x+(w-1)/2, y+(h-1)/2]), float(min(w,h)+2)))
    if len(posts) < 2:
        return
    post_tree = STRtree([Point(point) for point,_ in posts])
    nodes = {n['id']:n['point'] for n in graph['nodes']}
    edges = graph['edges']
    lines = [LineString([nodes[e['start']],nodes[e['end']]]) for e in edges]
    wall_tree = STRtree(lines)
    openings = [LineString(b.get('clear_span', b['span']))
                for b in graph['diagnostics'].get('opening_bridges', [])]
    scale = graph['scale_cm_per_px']
    for i,(a,width_a) in enumerate(posts):
        reach = min(8*width_a, 180/scale)
        for j in post_tree.query(box(*(a-reach), *(a+reach))):
            if j <= i:
                continue
            b,width_b = posts[j]
            width = min(width_a,width_b)
            length = np.linalg.norm(b-a)
            if (max(width_a,width_b) > 1.5*width or not 2*width < length <= 8*width
                    or not 60 <= length*scale <= 180):
                continue
            measured = raster_aperture(distance, a, b, width)
            if measured is None or narrow_frame_rails(gray, *measured, width):
                continue
            axis = (b-a)/length
            normal = np.array([-axis[1],axis[0]])
            flanks, endpoints, thicknesses = [], [], []
            for post, direction in ((a,-axis),(b,axis)):
                anchor = wall_anchor(post,direction,2*length,lines,wall_tree)
                if anchor is None:
                    break
                end,index = anchor
                if any(opening.distance(Point(end)) < 1 for opening in openings):
                    break
                thickness = edges[index]['thickness_px']
                start = post + direction*width/2
                finish = end - direction*thickness/2
                if (finish-start) @ direction < 8:
                    break
                rails = narrow_frame_rails(gray,start,finish,width)
                if not rails:
                    break
                endpoints.append(end.tolist())
                thicknesses.append(thickness)
                flanks.append(rails)
            if len(endpoints) != 2:
                continue
            # Each flank was sampled away from its post, so its local normal
            # points in the opposite direction to the other flank's normal.
            left, right = [f['rail_offsets_px'] for f in flanks]
            if max(abs(left[0]+right[-1]), abs(left[-1]+right[0])) > 1.5:
                continue
            leaves = jamb_face_leaves(gray, np.asarray(measured), normal, width)
            leaf = min((l for _,l in leaves), key=lambda l: l['jamb_distance_px'], default=None)
            if leaf:
                yield dict(partition_span=endpoints, door_span=np.asarray(measured).tolist(),
                           thickness_px=min(thicknesses), leaf=leaf, flanks=flanks)
