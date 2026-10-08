# Recognition ground truth

Accuracy is measured separately from export validity. A valid `project.json`
does not prove that rooms, walls or openings match the source image.

Ground-truth files live in `test/ground_truth/`. Coordinates are source-image
pixels before resize or letterbox. Every category declares its coverage:

- `count` — only the confirmed number of objects is evaluated;
- `geometry` — locations and shapes are evaluated;
- missing coverage — category is unannotated and excluded from scoring.

An empty array with `geometry` coverage means that the image is confirmed to
contain no objects of that category. It is different from an omitted category.

## Format

```json
{
  "version": 1,
  "image": {"file": "exmpls/plan.png", "width": 1200, "height": 800, "sha256": "..."},
  "coverage": {
    "rooms": "geometry",
    "walls": "geometry",
    "doors": "geometry",
    "windows": "geometry",
    "junctions": "geometry"
  },
  "tolerance": {
    "room_iou": 0.5,
    "opening_center_px": 16,
    "opening_angle_deg": 25,
    "opening_length_ratio": 0.45,
    "junction_center_px": 12
  },
  "rooms": [
    {"id": "r1", "polygon": [[20, 20], [500, 20], [500, 400], [20, 400]],
     "holes": [], "type": "enclosed"}
  ],
  "walls": [
    {"id": "wall-1", "centerline": [[20, 20], [500, 20]],
     "thickness_px": 18, "role": "exterior"}
  ],
  "doors": [
    {"id": "door-1", "span": [[240, 390], [320, 390]],
     "wall_id": "wall-2", "connects": ["r1", "outside"]}
  ],
  "windows": [
    {"id": "window-1", "span": [[100, 20], [220, 20]],
     "wall_id": "wall-1", "connects": ["r1", "outside"]}
  ],
  "junctions": [
    {"id": "junction-1", "type": "T", "point": [300, 200],
     "wall_ids": ["wall-3", "wall-4", "wall-5"]}
  ]
}
```

Walls use centerlines plus visible thickness. Doors and windows use the span in
the supporting wall, rather than a segmentation polygon. Junction type is `T`
or `X`; `wall_ids` records the incident annotated walls.

## Metrics

- rooms: one-to-one polygon matching, IoU, precision, recall and F1;
- walls: area precision/recall/IoU after buffering annotated centerlines by
  their thickness;
- doors/windows: center, orientation and length matching, scored separately so
  a window predicted as a door remains both a false door and a missed window;
- junctions: type and point distance. If recognition has no explicit junctions,
  they are derived from the wall-mask skeleton.

Run manually:

```powershell
python scripts/evaluate_recognition.py `
  test/ground_truth/clean-baseline.json `
  output/plan_rooms.json `
  --image exmpls/plan.png `
  --output output/plan_accuracy.json
```

`run_recognition_regression.py` runs the same evaluator when a manifest case
contains `ground_truth`. Its accuracy artifact is diagnostic: it does not
change `good`, `review` or `invalid` and cannot allow export.

Current coverage is intentionally limited. Source-image geometry was manually
traced by Codex on 2026-10-08 for four plans and has not been independently
reviewed:

| Plan | Geometric coverage | Initial observation |
|---|---|---|
| `clean-baseline` | 5 rooms, 5 doors, 6 windows | All matched; mean room IoU 0.9753. The counts were confirmed by the user. |
| `clean-diagonal` | 6 rooms, 6 doors, 7 windows | All matched; mean room IoU 0.9000. |
| `small-furnished` | 2 windows | 2 of 10 predictions matched; 8 false window candidates. |
| `colored-walls` | 5 windows | 3 of 8 predictions matched; 5 false candidates and 2 missed windows. |

Other categories on these plans are unannotated. In particular, walls and T/X
junctions have no geometric coverage. The evaluator checks that covered
geometry has valid IDs, in-bounds coordinates, nonempty room polygons, and
nonzero wall/opening spans before scoring. Opening and room metrics list
`unmatched_truth` IDs and zero-based `unmatched_predictions` indices; the
indices refer to the corresponding arrays in the room artifact.

These four examples are development cases, not a representative held-out set.
The furnished and colored examples confirm a substantial false-window problem.
The colored example also has missed real windows, so a stricter filter alone
could make recall worse. Its two missed top-right windows have no candidates
in the initial `openings` array, so they cannot be recovered by reclassifying
the existing candidates alone. One false positive is a narrow duplicate of
the correctly matched lower-right window; the others require separate visual
triage. Before tuning a new threshold,
annotate more distinct styles, keep some styles out of development, and compare
category-level false positives and false negatives on that held-out set.
Regression statuses and unchanged object counts cannot substitute for that
comparison.
