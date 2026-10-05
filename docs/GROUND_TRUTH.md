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
