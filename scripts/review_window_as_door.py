"""Apply image-bound, user-confirmed door type to existing window carriers."""
import argparse
import copy
import json
from pathlib import Path

from PIL import Image
from review_decisions import MAX_ANSWERS, image_sha256
from review_graph_passages import passage_fingerprint


def apply_window_door_reviews(graph, source, image_size, review, digest):
    if review.get('version') != 1 or review.get('image_sha256') != digest:
        raise ValueError('Door review image hash or format does not match')
    if graph.get('coordinate_space') != 'source_raster' or source.get('meta', {}).get('coordinate_space', 'source_raster') != 'source_raster':
        raise ValueError('Source-raster coordinates required')
    if (source.get('meta', {}).get('width'), source.get('meta', {}).get('height')) != image_size:
        raise ValueError('Source dimensions do not match image')
    answers = review.get('answers')
    if not isinstance(answers, list) or len(answers) > MAX_ANSWERS:
        raise ValueError('Invalid door review answers')
    result = copy.deepcopy(graph)
    seen = set()
    for answer in answers:
        fingerprint = answer.get('fingerprint') if isinstance(answer, dict) else None
        if not fingerprint or fingerprint in seen or answer.get('choice') != 'door' or not answer.get('user_statement'):
            raise ValueError('Unique explicit user door review required')
        seen.add(fingerprint)
        matches = [b for b in result['diagnostics']['opening_bridges'] if passage_fingerprint(b) == fingerprint]
        if len(matches) != 1:
            raise ValueError('Reviewed opening missing or changed')
        bridge = matches[0]
        if bridge.get('kind') == 'doors' and bridge.get('door_review', {}).get('image_sha256') == digest:
            continue
        if bridge.get('kind') != 'windows':
            raise ValueError('Expected a window candidate')
        if len([o for o in source.get('windows', []) if o.get('id') == bridge.get('opening_id')]) != 1:
            raise ValueError('Opening source is missing or ambiguous')
        bridge.update(kind='doors', requires_review=True,
                      door_review=dict(answer, image_sha256=digest, previous_kind='windows',
                                       basis='user_rule_and_visual_diagonal_line'))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('graph', type=Path); p.add_argument('source', type=Path); p.add_argument('image', type=Path)
    p.add_argument('--decisions', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); load = lambda path: json.loads(path.read_text(encoding='utf-8'))
    with Image.open(a.image) as im:
        result = apply_window_door_reviews(load(a.graph), load(a.source), im.size, load(a.decisions), image_sha256(a.image))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print('Reviewed door choices applied')


if __name__ == '__main__':
    main()
