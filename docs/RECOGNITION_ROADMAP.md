# Recognition Roadmap

## Baseline — 2026-10-02

CPU-запуск подготовлен, эталон `201535.png` воспроизведён: 5 rooms / 5 doors / 6 windows.
Все 10 примеров прогнаны: 5 сформировали JSON, 5 вернули 0 rooms и остановились при построении проекта.
Это техническая успешность, не accuracy. Требования, окружение, диагностика и команды повторного запуска — [REGRESSION_BASELINE.md](REGRESSION_BASELINE.md).
Preprocessing V2 и room extraction V2 реализованы под `--pipeline-v2`: эталон сохранён, техническая успешность составила 8/10, V1 не изменился. Ограничения и проверки — [PIPELINE_V2.md](PIPELINE_V2.md).
Следующая итерация — quality gate; V2 остаётся экспериментальным.

## Цель

Сделать `flat-recognize` устойчивым к реальным пользовательским изображениям:
- скриншотам;
- сканам;
- мебели;
- размерным линиям;
- тексту;
- логотипам/watermark;
- серым, чёрным и цветным стенам;
- разной толщине линий;
- большим полям;
- балконам и дополнительным контурам.

Рабочую интеграцию в `project.json` сохраняем.

## Phase 1 — Robust input preprocessing

Новый модуль:

```text
scripts/preprocess_v2.py
```

Выход:
- normalized image;
- ROI coordinates;
- diagnostic preview;
- transform metadata.

Что делать:
1. убрать очевидные внешние поля;
2. определить dominant floorplan region;
3. не вырезать детали внутри квартиры;
4. сохранить mapping crop → source;
5. подготовить original/grayscale/contrast-normalized representations.

Не полагаться только на правило "широкая чёрная полоса → crop".

Предпочтительно:

```text
whole image
 → coarse structural prediction / line density
 → connected structural regions
 → choose dominant floorplan ROI
 → padded crop
```

Acceptance:
- VectorStock footer;
- большой логотип;
- текст вне плана;
- размерные подписи вокруг здания;
- большие белые поля.

## Phase 2 — `extract_rooms_v2`

Проблема текущего подхода:

```text
free-space touching border = outside
```

Один leak может удалить все комнаты.

Новый подход:

```text
wall mask
 +
known doors/windows
 +
temporary inferred gap closures
        ↓
room barrier
        ↓
room candidates
```

Inferred closure используется только для topology и не становится настоящей стеной.

Шаги:
1. barrier из walls;
2. закрыть detected openings;
3. найти wall endpoints;
4. найти соосные пары endpoints с разумным gap;
5. временно закрыть gaps;
6. connected components;
7. определить outside;
8. построить polygons;
9. проверить topology.

Adaptive passes:
- pass 1 — conservative;
- pass 2 — если `rooms == 0` или topology плохая, слегка расширить gap/morphology.

Diagnostics:

```text
Rooms pass 1: 0
Candidate gaps: 5
Temporary closures: 3
Rooms pass 2: 4
```

## Phase 3 — Quality Gate

Новый модуль:

```text
scripts/validate_recognition.py
```

Проверки:
- есть rooms;
- room polygons valid;
- нет massive overlap;
- segments/connectors valid;
- нет zero-length segments;
- areas замкнуты;
- doors/windows ссылаются на существующую wall;
- distance-to-wall допустим;
- opening position внутри segment;
- нет очевидного room leak.

Результат:

```json
{
  "quality": 0.87,
  "status": "good",
  "warnings": []
}
```

или:

```json
{
  "quality": 0.44,
  "status": "review",
  "warnings": [
    "2 rooms have uncertain closure",
    "1 opening is far from any wall"
  ]
}
```

UI должен показывать предупреждение, а не всегда безусловный success.

## Phase 4 — Dual-pass inference

Pass A:
```text
original / lightly normalized
```

Pass B:
```text
grayscale + contrast normalization
```

Начальная fusion-стратегия:
- wall probability — больший вес normalized pass;
- door/window probability — больший вес original pass.

Любой fusion должен проходить clean regression baseline.

## Phase 5 — Regression dataset

Структура:

```text
tests/recognition_cases/
    clean/
    furniture/
    dimensions/
    colored/
    scans/
    watermarks/
    difficult_openings/
```

Для кейса хранить expectation metadata:

```json
{
  "expected_rooms": 5,
  "expected_doors": 4,
  "expected_windows": 6,
  "notes": "dimension lines around outer contour"
}
```

Минимальные regression metrics:
- rooms count;
- doors count;
- windows count;
- pipeline success/fail;
- quality score.

Позже добавить polygon IoU.

## Phase 6 — Synthetic augmentation

Главный путь к универсальности текущей модели.

Берём clean training sample и сохраняем segmentation target неизменным, а поверх input генерируем:
- цифры;
- площади `4,9 м²`;
- room labels;
- русский/английский текст;
- dimension lines/arrows/ticks;
- мебельные примитивы;
- watermark/logo/footer;
- gray/brown/black walls;
- low contrast;
- scan noise;
- blur;
- JPEG compression;
- large margins;
- небольшие rotation/perspective distortions.

Главное правило:

```text
overlay меняет input only
segmentation mask остаётся ground truth
```

Модель должна выучить:

```text
furniture/text/dimensions != structure
```

## Phase 7 — Fine-tune current model

Сначала дообучать текущую архитектуру, потому что:
- CPU pipeline уже работает;
- integration готова;
- postprocessing готов.

Training strategy:
1. baseline existing weights;
2. mix clean + augmented samples;
3. не обучать только на noisy;
4. отслеживать wall IoU, opening recall/precision, room topology success.

Главная end-to-end метрика:

```text
image → valid editable project.json
```

## Phase 8 — Scale calibration

Поддержать:
- manual known distance;
- wall-thickness calibration;
- в будущем dimension OCR.

OCR не должен быть обязательным путём. Если confidence низкий — не использовать.

## Phase 9 — More general geometry

Уйти от room bbox assumptions к general wall graph.

Поддержать:
- L-shape;
- arbitrary polygons;
- diagonal walls;
- variable wall thickness;
- complex T/X junctions.

Желаемый переход:

```text
room bbox assumptions
        ↓
centerlines / skeleton
        ↓
junction detection
        ↓
general wall graph
```

## Phase 10 — Alternative model benchmark

Только после появления собственного regression set.

Кандидаты:
- RoomFormer;
- Raster2Seq;
- другие polygon/topology-first models.

Сравнивать:
- CPU/GPU requirement;
- latency;
- wall quality;
- room topology success;
- door/window accuracy;
- final JSON success rate.

Текущий CPU-compatible pipeline остаётся baseline.

## Приоритет ближайших задач

P0:
```text
[x] preprocess_v2.py (experimental, --pipeline-v2)
[x] extract_rooms_v2.py (experimental, --pipeline-v2)
[ ] quality gate
[x] regression cases (10 images, confirmed expectations for clean baseline)
```

P1:
```text
[ ] dual-pass inference
[ ] better wall graph
[ ] scale calibration
```

P2:
```text
[ ] synthetic augmentation generator
[ ] fine-tune
[ ] alternative model benchmark
```

## Что не делать сейчас

Не тратить время на:
- production-polish UI;
- furniture recognition;
- 3D furniture generation;
- semantic room naming;
- OCR всех надписей;
- замену всего pipeline новой моделью без benchmark.

Сначала добиться устойчивого:

```text
walls
rooms
doors
windows
```

на разнообразных пользовательских изображениях.

## Рекомендуемый следующий implementation task

Реализовать:

```text
scripts/preprocess_v2.py
scripts/extract_rooms_v2.py
```

и встроить feature flag:

```bash
python scripts/process_floorplan.py input.jpg --pipeline-v2
```

Пока V2 не пройдёт regression set, V1 оставить рабочим. После стабилизации V2 переключить на default.
