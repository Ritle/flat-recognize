"""Record explicit image-bound confirmations of existing graph windows.

No geometry or opening type is inferred or altered. This is separate from
window-to-door and open-passage review; exterior windows can also be confirmed.
"""
import argparse
import copy
import json
from pathlib import Path

from PIL import Image
from review_decisions import MAX_ANSWERS, image_sha256
from review_graph_passages import passage_fingerprint


def confirm_graph_windows(graph, source, image_size, review, digest):
    if review.get('version') != 1 or review.get('image_sha256') != digest:
        raise ValueError('Window review image hash or format does not match')
    if (graph.get('coordinate_space') != 'source_raster'
            or source.get('meta', {}).get('coordinate_space', 'source_raster') != 'source_raster'
            or (source.get('meta', {}).get('width'), source.get('meta', {}).get('height')) != image_size):
        raise ValueError('Source-raster coordinates and image dimensions required')
    answers = review.get('answers')
    if not isinstance(answers, list) or len(answers) > MAX_ANSWERS:
        raise ValueError('Invalid window review answers')
    result = copy.deepcopy(graph)
    seen = set()
    for answer in answers:
        if not isinstance(answer, dict):
            raise ValueError('Invalid window review answer')
        fingerprint, statement = answer.get('fingerprint'), answer.get('user_statement')
        if (not isinstance(fingerprint, str) or not fingerprint or fingerprint in seen
                or answer.get('choice') != 'window' or not isinstance(statement, str) or not statement.strip()):
            raise ValueError('Unique explicit window choice and user statement required')
        seen.add(fingerprint)
        matches = [b for b in result.get('diagnostics', {}).get('opening_bridges', [])
                   if passage_fingerprint(b) == fingerprint]
        if len(matches) != 1 or matches[0].get('kind') not in ('windows', 'recovered_window'):
            raise ValueError('Reviewed window is missing, ambiguous or changed')
        matches[0]['window_review'] = dict(image_sha256=digest, choice='window',
                                            user_statement=statement, reviewer='user')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('graph', type=Path);p.add_argument('source', type=Path);p.add_argument('image', type=Path)
    p.add_argument('--decisions', type=Path, required=True);p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.resolve() in {x.resolve() for x in (a.graph,a.source,a.image,a.decisions)}:
        raise ValueError('Output must not overwrite inputs')
    load = lambda path: json.loads(path.read_text(encoding='utf8'))
    with Image.open(a.image) as image:
        size = image.size
    result = confirm_graph_windows(load(a.graph), load(a.source), size, load(a.decisions), image_sha256(a.image))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
    print(json.dumps({'confirmed': len(load(a.decisions)['answers'])}))


if __name__ == '__main__':main()
