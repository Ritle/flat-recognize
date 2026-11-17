"""Make a visual contact sheet of serialized graph openings outside partial GT coverage.

The sheet is diagnostic only: it does not determine whether an object is valid,
change predictions, or use reference labels during recognition.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from evaluate_graph_export import graph_export_geometry


def outside_regions(span, regions):
    x = (span[0][0] + span[1][0]) / 2
    y = (span[0][1] + span[1][1]) / 2
    return not any(x1 <= x <= x2 and y1 <= y <= y2 for x1, y1, x2, y2 in regions)


def review_items(geometry, ground_truth=None):
    """Use the evaluator's center-in-box rule to identify unscored openings."""
    rows = []
    for kind in ('doors', 'windows', 'passages'):
        coverage = (ground_truth or {}).get('coverage', {}).get(kind)
        regions = (ground_truth or {}).get('annotation_regions', {}).get(kind) if coverage == 'geometry' else None
        for index, opening in enumerate(geometry[kind], 1):
            if coverage == 'geometry' and not regions:
                continue  # Geometry GT without regions covers the entire image.
            if regions and not outside_regions(opening['span'], regions):
                continue
            rows.append(dict(kind=kind, number=index, uuid=opening['id'], span=opening['span'],
                             coverage='outside_regions' if regions else 'unannotated'))
    return rows


def render_contacts(image, items, output_dir):
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError('Contact sheet directory must be empty')
    output_dir.mkdir(parents=True, exist_ok=True)
    # Two panes preserve original evidence alongside an exact export-hole overlay.
    columns, page_size, cell_w, cell_h = 3, 12, 340, 175
    for start in range(0, len(items), page_size):
        subset = items[start:start + page_size]
        sheet = Image.new('RGB', (columns * cell_w, math.ceil(len(subset)/columns)*cell_h), 'white')
        draw = ImageDraw.Draw(sheet)
        for offset, item in enumerate(subset):
            x0, y0 = (offset % columns)*cell_w, (offset // columns)*cell_h
            span = item['span']; x = sum(p[0] for p in span)/2; y = sum(p[1] for p in span)/2
            radius = max(30, math.dist(*span)*.8 + 15)
            box = (max(0, math.floor(x-radius)), max(0, math.floor(y-radius)),
                   min(image.width, math.ceil(x+radius)), min(image.height, math.ceil(y+radius)))
            if box[0] >= box[2] or box[1] >= box[3]:
                raise ValueError('Opening is outside the source image')
            raw = image.crop(box).convert('RGB'); marked = raw.copy()
            ImageDraw.Draw(marked).line([(p[0]-box[0], p[1]-box[1]) for p in span], fill='magenta', width=2)
            for pane, source in enumerate((raw, marked)):
                fitted = ImageOps.contain(source, (162, 142), Image.Resampling.NEAREST)
                sheet.paste(fitted, (x0 + 3 + pane*168 + (162-fitted.width)//2,
                                     y0 + 27 + (142-fitted.height)//2))
            draw.text((x0+4,y0+4), f"{item['kind']} #{item['number']} | original / hole", fill='black')
            item.update(page=start//page_size+1, cell=offset+1, crop=list(box))
        sheet.save(output_dir / f'contacts-{start//page_size+1:02d}.png')
    (output_dir/'contacts.json').write_text(json.dumps(items, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('image', type=Path);p.add_argument('project', type=Path);p.add_argument('report', type=Path)
    p.add_argument('--ground-truth', type=Path, help='Optional partial GT used only to select unscored items')
    p.add_argument('--output-dir', type=Path, required=True)
    a=p.parse_args();load=lambda path:json.loads(path.read_text(encoding='utf8'))
    truth=load(a.ground_truth) if a.ground_truth else None
    expected_hash = (truth or {}).get('image', {}).get('sha256')
    if expected_hash and expected_hash != hashlib.sha256(a.image.read_bytes()).hexdigest():
        raise ValueError('Ground truth image hash does not match')
    with Image.open(a.image) as image:
        if truth and (truth['image']['width'],truth['image']['height']) != image.size:
            raise ValueError('Ground truth image size does not match')
        geometry=graph_export_geometry(load(a.project),load(a.report),{'width':image.width,'height':image.height})
        items=review_items(geometry,truth)
        render_contacts(image,items,a.output_dir)
    print(json.dumps({'items':len(items),'doors':sum(i['kind']=='doors' for i in items),
                      'windows':sum(i['kind']=='windows' for i in items),
                      'passages':sum(i['kind']=='passages' for i in items)}))


if __name__=='__main__':main()
