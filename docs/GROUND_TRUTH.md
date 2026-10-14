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

For a complex image, `annotation_regions` may restrict `doors` or `windows`
geometry to fully inspected source-image rectangles `[x1, y1, x2, y2]`. Only
predictions whose opening-span centers fall in those rectangles are scored.
Every annotated span center must also lie inside its category's rectangles.
Metrics label this as `scope: annotation_regions`, record original prediction
indices, and never claim accuracy for the rest of the image. Do not place a
region boundary through an opening whose status is uncertain.

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
traced by Codex on 2026-10-08, 2026-10-09 and 2026-10-14 for nine plans and has not been
independently reviewed:

| Plan | Geometric coverage | Initial observation |
|---|---|---|
| `clean-baseline` | 5 rooms, 5 doors, 6 windows | All matched; mean room IoU 0.9753. The counts were confirmed by the user. |
| `clean-diagonal` | 6 rooms, 6 doors, 7 windows | All matched; mean room IoU 0.9000. |
| `small-furnished` | 2 windows | 2 of 10 predictions matched; 8 false window candidates. |
| `colored-walls` | 5 windows | Initially 3 of 8 matched (5 false, 2 missed); after edge-strip deduplication 3 of 7 matched (4 false, 2 missed). |
| `open-zones` | 2 doors and 6 top-facade windows, scoped | Doors 2/0/0; windows improved from 5/5/1 to 5/1/1 TP/FP/FN inside annotated regions after perpendicular-endcap filtering. |
| `complex-hatched` | 2 central doors and 4 side-facade windows, scoped | Doors 2/0/0; windows improved from 2/0/2 to 4/0/0 TP/FP/FN inside annotated regions after weak-arc reclassification. |
| `heavy-black-walls` | 1 bathroom door and 2 north-facade windows, scoped | Door 1/0/0; windows 2/2/0 initially and 2/0/0 after one-sided candidate review. |
| `dense-hatched-multizone` | 1 northwest door and 3 west-facade windows, scoped | Door 0/0/1; windows 3/0/0. Other densely hatched rooms and openings are unannotated. |
| `l-shaped-diagonal-bath` | 2 doors and 3 north-facade windows, scoped | Doors 2/0/0; windows 3/0/0. The balcony and diagonal bathroom opening are unannotated. |

On `open-zones`, four unsupported vertical window candidates cross longer,
wall-supported horizontal windows. The filter records their IDs and parent
windows, removes them from window export, and preserves their polygons in
`reclassified_wall_support` for explicit boundary support. Room topology is
rechecked first. This prevents deleting the only evidence for three existing
room-boundary segments; it does not certify those segments as accurate walls.
The project remains `invalid` under the final quality gate.

On `complex-hatched`, the two upper side windows were segmented as opening
polygons but misclassified as doors because nearby furniture lines weakly
resembled swing arcs. They are reclassified as review-required windows only
when a long weak-arc candidate aligns with a separate, similar-sized,
wall-supported window bay. This does not affect room topology and does not
resolve the remaining final-project quality errors.

Other categories on these plans are unannotated. In particular, walls and T/X
junctions have no geometric coverage. The evaluator checks that covered
geometry has valid IDs, in-bounds coordinates, nonempty room polygons, and
nonzero wall/opening spans before scoring. Opening and room metrics list
`unmatched_truth` IDs and zero-based `unmatched_predictions` indices; the
indices refer to the corresponding arrays in the room artifact.

These six examples are development cases, not a representative held-out set.
The furnished and colored examples confirm a substantial false-window problem.
The colored example also has missed real windows, so a stricter filter alone
could make recall worse. Its two missed top-right windows have no candidates
in the initial `openings` array, so they cannot be recovered by reclassifying
the existing candidates alone. One false positive was a narrow duplicate of
the correctly matched lower-right window and is now rejected with a recorded
parent candidate; the others require separate visual triage. Before tuning a new threshold,
annotate more distinct styles, keep some styles out of development, and compare
category-level false positives and false negatives on that held-out set.
Regression statuses and unchanged object counts cannot substitute for that
comparison.
