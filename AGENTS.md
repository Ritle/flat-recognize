# AGENTS.md

## Назначение проекта

`flat-recognize` — экспериментальный сервис распознавания растровых планировок квартир и преобразования результата в JSON-формат пользовательского приложения.

Основная цель текущего MVP:

```text
PNG / JPG / WEBP
        ↓
структурное распознавание
        ↓
walls / rooms / doors / windows
        ↓
connectors / segments / areas / elements
        ↓
готовый project.json
```

Итоговый JSON уже успешно открывается в целевом приложении.

## Главные ограничения и принципы

1. Не ломать уже работающий экспорт JSON. Формат `connectors / segments / areas / elements` — интеграционный контракт.
2. CPU inference должен оставаться рабочим. GPU может поддерживаться дополнительно, но не становиться обязательным для MVP.
3. Не переписывать рабочие этапы без необходимости. Сначала локализовать проблему по промежуточным JSON и overlay.
4. Сохранять диагностические артефакты: segmentation overlay, classified openings overlay, room overlay, barrier mask, intermediate JSON.
5. Не хранить в Git `.venv/`, `weights/`, `*.safetensors`, `output/`, `demo_jobs/`, пользовательские изображения и runtime logs.
6. Не менять лицензионную стратегию без отдельной задачи. Сейчас существующие веса сохраняются для разработки и PoC.
7. Не добавлять эвристику только ради одного изображения, если она ухудшает другие типы планировок.

## Текущая архитектура

База — `floorplan-to-3d`:
- ResNet34 + U-Net-style segmentation;
- классы `floor`, `wall`, `door`, `window`;
- CPU inference;
- веса скачиваются отдельно и не хранятся в Git.

Основной код модели — `src/buildingcv/`.

## Текущий pipeline

### 1. `scripts/predict_raster.py`

Вход: PNG/JPG/WEBP.

Делает:
- загрузку raster image;
- aspect-preserving resize / letterbox;
- нормализацию;
- inference;
- `argmax` mask;
- polygon extraction;
- возврат координат в исходное изображение;
- фильтрацию opening-кандидатов;
- диагностический overlay.

Структурный результат:

```json
{
  "meta": {},
  "walls": [],
  "openings": []
}
```

Semantic class `door/window` от модели используется как дополнительный сигнал, а не как окончательная истина.

### 2. `scripts/classify_openings.py`

Назначение:

```text
openings → doors + windows
```

Текущий метод:
- grayscale/Canny;
- HoughLinesP;
- поиск линии дверного полотна;
- при наличии характерной геометрии → `door`;
- иначе → `window`.

На чистом regression-кейсе получено:

```text
5 doors
6 windows
```

### 3. `scripts/extract_rooms.py`

Текущий метод:
1. barrier mask из walls;
2. временное закрытие doors/windows;
3. morphology для мелких разрывов;
4. invert → free space;
5. `connectedComponentsWithStats`;
6. компоненты, касающиеся края изображения, считаются outside;
7. остальные крупные компоненты считаются rooms;
8. строятся room polygons.

На чистом тесте:

```text
5 rooms
```

Известная слабость: если реальный проём остаётся незакрытым, interior free-space может соединиться с внешним фоном. Тогда текущая логика способна вернуть `rooms: 0`.

### 4. `scripts/build_project_json.py`

Назначение:

```text
rooms → project geometry
```

Создаёт:
- `connectors`;
- `adjacency`;
- `segments`;
- `areas`.

Текущая версия лучше всего работает на ортогональных планировках.

Базовые параметры:
- wall height: 270 cm;
- wall thickness: 20 cm;
- scale: configurable.

Полученный JSON уже успешно открывается в целевом приложении.

### 5. `scripts/add_openings.py`

Назначение:
- найти ближайший segment для каждой двери;
- вычислить position вдоль стены;
- создать `doorItem`;
- сформировать bindings для окон.

Ключевые поля:
- `wall.uuid`;
- `wall.position`;
- `size`;
- `holeShape`;
- `model`;
- `scale`;
- `materials`.

Текущая поправка ширины двери:

```text
recognized_width × 1.12
```

Важный invariant:

```text
materials[*].itemUuid == door.uuid
```

При клонировании template нельзя оставлять `itemUuid` исходной шаблонной двери.

### 6. `scripts/add_windows.py`

Назначение:
- использовать `window_bindings`;
- клонировать реальный `windowItem` из template;
- привязать окно к segment;
- масштабировать width, holeShape, sill и model scale.

### 7. `scripts/process_floorplan.py`

Главный CLI orchestration layer.

Целевая команда:

```bash
python scripts/process_floorplan.py path/to/floorplan.jpg
```

Pipeline:

```text
predict_raster
    ↓
classify_openings
    ↓
extract_rooms
    ↓
build_project_json
    ↓
add_openings
    ↓
add_windows
    ↓
final project.json
```

Не дублировать recognition-алгоритмы внутри `process_floorplan.py`.

## Web demo

### `demo/app.py`

FastAPI endpoint:

```http
POST /api/recognize
```

Функции:
- загрузка изображения;
- запуск pipeline;
- отдельный job;
- статистика;
- выдача JSON;
- скачивание `project.json`;
- diagnostic images.

Runtime jobs:

```text
demo_jobs/<job-id>/
```

### `demo/index.html`

UI:
- drag & drop;
- JPG/PNG/WEBP;
- статистика rooms/walls/doors/windows;
- overlays;
- JSON preview;
- download `project.json`.

На текущем сервере Apache использует `8080`, поэтому dev-demo запускалась на `8081`.

## Templates

Используются как референс сериализации:

```text
test/template.json
test/template_window.json
```

Не удалять их без замены на программно определённую schema/defaults.

## Подтверждённый regression baseline

На чистом тестовом плане:

```text
rooms:   5
doors:   5
windows: 6
```

Стены/комнаты/двери/окна импортированы в целевое приложение.

Любое изменение распознавания не должно ломать этот кейс.

## Известные проблемы

### P0 — room leakage

Типичная цепочка:

```text
незакрытый проём
→ interior free space соединяется с outside
→ component касается image border
→ component удаляется
→ rooms = 0
→ build_project_json.py: No rooms in source JSON
```

Исправлять сначала room extraction/preprocessing, а не маскировать ошибку в `build_project_json.py`.

### P0 — domain shift

Текущая модель хуже работает на:
- мебели;
- размерных линиях;
- цифрах и подписях;
- логотипах;
- watermark;
- цветных/серых стенах;
- штриховках;
- сканах;
- JPEG artifacts;
- разных стилях дверей/окон;
- больших полях вокруг плана.

Не решать это исключительно удалением тонких линий OpenCV: настоящие архитектурные элементы тоже могут быть тонкими.

### P1 — geometry assumptions

Отдельно тестировать:
- L-shaped rooms;
- non-rectangular rooms;
- диагональные стены;
- variable wall thickness;
- сложные junctions.

### P1 — absolute scale

Raster не гарантирует physical scale. Поддерживать:
- явный `--scale`;
- calibration по wall thickness;
- в будущем — calibration по известной длине/размерным надписям.

## Следующий приоритет

```text
1. robust preprocessing / ROI
2. extract_rooms_v2
3. quality gate
4. dual-pass inference
5. regression dataset
6. synthetic augmentation
7. fine-tune
8. benchmark альтернативных моделей
```

Подробности — `docs/RECOGNITION_ROADMAP.md`.

## Regression workflow для Codex

Перед изменением recognition:
1. определить failing stage;
2. изучить intermediate JSON;
3. изучить overlay;
4. сделать минимальное изменение;
5. прогнать clean regression case;
6. прогнать несколько сложных планов;
7. проверить итоговый JSON в приложении;
8. только затем менять следующий слой.

Не смешивать в одном изменении model inference + room extraction + target JSON schema, если это не необходимо.

## Definition of Done для MVP recognition

- пользователь загружает JPG/PNG/WEBP;
- типичные поля/логотипы не ломают crop;
- furniture/text редко превращаются в walls;
- room extraction не падает из-за одного незакрытого doorway;
- каждый opening привязан к существующему segment;
- final JSON валиден;
- final JSON открывается в целевом приложении;
- сервис сообщает о низкой уверенности вместо молчаливой генерации явно плохого проекта;
- CPU deployment остаётся возможным.
