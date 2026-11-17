"""Render actual serialized graph walls and opening cuts over the source image."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw
from evaluate_graph_export import graph_export_geometry


def render_graph_export(image, project, report):
    geometry = graph_export_geometry(project, report, {'width': image.width, 'height': image.height})
    transform = report['pixel_to_world']; scale = transform['scale']; ox, oy = transform['origin']
    level = project['levels'][0]
    points = {c['uuid']: ((c['x'] - ox) / scale, (c['y'] - oy) / scale) for c in level['connectors']}
    overlay = image.convert('RGB').copy()
    draw = ImageDraw.Draw(overlay)
    for segment in level['segments']:
        draw.line([points[segment['start']], points[segment['end']]], fill='yellow', width=2)
    colors = {'doors': '#00e070', 'windows': '#00bfff', 'passages': '#ff00cf'}
    for kind, color in colors.items():
        for index, opening in enumerate(geometry[kind], 1):
            a, b = opening['span']
            draw.line([tuple(a), tuple(b)], fill=color, width=5)
            midpoint = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
            draw.text((midpoint[0] + 3, midpoint[1] + 3), str(index), fill=color,
                      stroke_width=1, stroke_fill='black')
    return overlay, geometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path); parser.add_argument('project', type=Path)
    parser.add_argument('report', type=Path); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    metadata = args.output.with_suffix('.json')
    if args.output.suffix.lower() != '.png' or {args.output.resolve(), metadata.resolve()} & {
            p.resolve() for p in (args.image, args.project, args.report)}:
        raise ValueError('Expected a PNG output that does not overwrite inputs')
    load = lambda path: json.loads(path.read_text(encoding='utf-8'))
    with Image.open(args.image) as image:
        overlay, geometry = render_graph_export(image, load(args.project), load(args.report))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(args.output)
    metadata.write_text(json.dumps(geometry, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({kind: len(geometry[kind]) for kind in ('rooms', 'doors', 'windows', 'passages')}))


if __name__ == '__main__':
    main()
