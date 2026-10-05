# Development Map

`polygon_geometry.py` добавляет общий граф границ сложных комнат. V2 выбирает
его автоматически; неподтверждённые границы блокируют экспорт до переноса
проёмов. Детали: [POLYGON_GEOMETRY.md](POLYGON_GEOMETRY.md).

Для V1/V2 добавлены `validate_recognition.py`, отчёт качества и блокировка
некорректного экспорта в demo. Контракт и проверки: [QUALITY_GATE.md](QUALITY_GATE.md).
V2 запускает выровненные исходный и нормализованный проходы модели;
`dual_pass_inference.py` консервативно объединяет вероятности и сохраняет три
диагностические маски. Детали: [PIPELINE_V2.md](PIPELINE_V2.md).
После room extraction V2 запускает `recover_openings_v2.py`: проверяет принятые
разрывы по стенам, полотну и дуге, сохраняет отдельные результаты для адаптера.
Детали: [OPENING_RECOVERY.md](OPENING_RECOVERY.md).
`door_leaf_evidence.py` даёт общую привязку полотна к стеновому торцу для
room repair и последующего подтверждения физической двери по дуге.
В V2 адаптер дополнительно запускает `internal_wall_graph.py`: переносит
подтверждённые маской внутренние перегородки, создаёт общие T/X-узлы и
обновляет cycles комнат. Детали: [INTERNAL_WALL_GRAPH.md](INTERNAL_WALL_GRAPH.md).

## End-to-end flow

```text
User image
   │
   ▼
predict_raster.py
   │
   ├── wall polygons
   └── opening polygons
   │
   ▼
classify_openings.py
   │
   ├── doors
   └── windows
   │
   ▼
extract_rooms.py
   │
   └── room polygons
   │
   ▼
build_project_json.py
   │
   ├── connectors
   ├── segments
   └── areas
   │
   ▼
add_openings.py
   │
   ├── doorItem
   └── window bindings
   │
   ▼
add_windows.py
   │
   └── windowItem
   │
   ▼
project.json
```

`process_floorplan.py` запускает эти этапы последовательно.

При `--pipeline-v2` перед inference запускается `preprocess_v2.py`, координаты возвращаются в исходный растр, а `extract_rooms.py` заменяется на `extract_rooms_v2.py`. V1 остаётся default. Реализация и ограничения: [PIPELINE_V2.md](PIPELINE_V2.md).

## Source map

```text
flat-recognize/
│
├── src/buildingcv/
│   ├── model.py
│   ├── extract_polygons.py
│   └── training/inference support
│
├── scripts/
│   ├── predict_raster.py
│   ├── dual_pass_inference.py
│   ├── classify_openings.py
│   ├── extract_rooms.py
│   ├── build_project_json.py
│   ├── add_openings.py
│   ├── add_windows.py
│   ├── process_floorplan.py
│   ├── preprocess_v2.py
│   ├── extract_rooms_v2.py
│   ├── recover_openings_v2.py
│   ├── door_leaf_evidence.py
│   ├── internal_wall_graph.py
│   ├── run_recognition_regression.py
│   └── auto_crop_plan.py
│
├── demo/
│   ├── app.py
│   └── index.html
│
├── test/
│   ├── template.json
│   └── template_window.json
│
├── weights/            # runtime only, ignored by Git
├── output/             # generated, ignored by Git
├── demo_jobs/          # generated, ignored by Git
│
├── AGENTS.md
└── docs/
    ├── DEVELOPMENT_MAP.md
    └── RECOGNITION_ROADMAP.md
```

## Responsibility boundaries

### Model layer

```text
src/buildingcv/
predict_raster.py
```

Ответственность:
- segmentation;
- polygon extraction;
- initial opening geometry.

Не должен знать JSON schema конечного приложения.

### Structural postprocessing

```text
classify_openings.py
extract_rooms.py
```

Ответственность:
- door/window semantics;
- room topology;
- repair uncertain masks.

Не должен знать 3D model URLs/materials конечного приложения.

### Project adapter

```text
build_project_json.py
add_openings.py
add_windows.py
```

Ответственность:
- target JSON;
- connectors/segments/areas;
- wall attachment;
- template-based elements.

Не должен выполнять neural inference.

### Orchestration

```text
process_floorplan.py
```

Ответственность:
- порядок этапов;
- paths;
- CLI params;
- final validation.

Не переносить сюда recognition logic.

### Demo/API

```text
demo/app.py
demo/index.html
```

Ответственность:
- upload;
- job lifecycle;
- diagnostics;
- download final JSON.

Не делать recognition logic в FastAPI handler.

## Data contracts

### Coordinates

До `build_project_json.py` координаты относятся к raster space.

После конвертации — project coordinate system.

Любой crop/preprocessing должен сохранять явный transform между пространствами координат.

### Openings

Каждый opening должен давать:
- center;
- width;
- orientation;
- nearest wall;
- position on wall.

### Rooms

Room polygon должен быть:
- не пустой;
- валидный;
- достаточно крупный;
- отделён от outside либо помечен uncertain.

### Final elements

Каждый `doorItem/windowItem` должен ссылаться на существующий `wall.uuid`.

## Current baseline parameters

```text
wall thickness: 20 cm
wall height: 270 cm
door width factor: 1.12
CPU inference: required
```

Window height/sill берутся из application template, если не overridden.

## Current known failure chain

```text
complex plan
   ↓
model misses opening / wall fragment
   ↓
room barrier leaks
   ↓
interior connects to outside
   ↓
connected component touches border
   ↓
rooms=[]
   ↓
build_project_json.py raises:
No rooms in source JSON
```

Правильная точка исправления: preprocessing / segmentation / room extraction.

## Current milestone

Подтверждено:

```text
clean floorplan
→ 5 rooms
→ 5 doors
→ 6 windows
→ valid project JSON
→ successful import into target application
```

Следующий milestone:

```text
mixed real-world floorplans
→ stable ROI
→ no room leakage
→ quality score
→ same valid project JSON
```
