# Правила разработки flat-recognize (для Codex и участников)

Срез 12.11.2026. Перед работой прочитайте [README.md](README.md), [текущее состояние](docs/CURRENT_STATUS.md) и [карту модулей](docs/DEVELOPMENT_MAP.md). Не считать устаревшие списки задач доказательством отсутствия функции.

## Цель и интеграционный контракт

Вход: растровый PNG/JPG/WEBP. Выход: project.json целевого редактора с уровнями, connectors, segments, areas и elements (doorItem/windowItem). test/template.json и test/template_window.json — шаблоны совместимости. Не менять структуру полей, UUID-связей, привязки openings и материалы без отдельной задачи и тестов. В частности, materials[*].itemUuid должен совпадать с UUID созданного элемента.

Сегментация и статистика по маскам — не доказательство корректного project.json. Изменения распознавания обязательно проверять сквозным экспортом, валидатором и регрессией.

## Что уже реализовано

- scripts/process_floorplan.py — единый CLI orchestrator; V1 по умолчанию, V2 включается --pipeline-v2.
- scripts/predict_raster.py, classify_openings.py — модель и проверка кандидатов дверей/окон.
- extract_rooms.py — V1; preprocess_v2.py, extract_rooms_v2.py, recover_windows_topology_v2.py, recover_openings_v2.py, filter_interior_windows_v2.py, reconcile_topology_windows_v2.py — V2 топология.
- build_project_json.py, polygon_geometry.py, internal_wall_graph.py, add_openings.py, add_windows.py — геометрия и сериализация.
- validate_recognition.py — обязательный gate для комнаты, геометрии, привязок, итогового проекта.
- review_decisions.py, review_rooms.py, review_partitions.py, detect_exterior_enclosures_v2.py, review_exterior_enclosures.py — ручное подтверждение ограниченного набора спорных объектов с валидацией исходного изображения.
- demo/app.py и demo/index.html — растровое demo; server/main.py и viewer/ — отдельный унаследованный SVG→3D сценарий.
- src/buildingcv/ — обучение и inference; scripts/compare_training_baseline.py — протокол отбора кандидата весов.

## Непереговорные ограничения

1. CPU обязателен для MVP. Не вводить зависимость от CUDA как условие запуска.
2. V1 остаётся поведением по умолчанию; V2 — явный экспериментальный режим. Не менять default молча.
3. Не обходить quality gate, не выдавать invalid за допустимый JSON и не путать quality/status с accuracy.
4. Привязки дверей/окон должны ссылаться на реальный segment, быть геометрически допустимыми; общие вершины и границы помещений должны оставаться согласованными.
5. Не создавать ложную архитектуру без подтверждения исходной маской/растром. При неоднозначности — вопрос пользователю или invalid, а не оптимистичный экспорт.
6. Координаты V2 после preprocess/dual-pass должны соответствовать исходному изображению. --scale задаёт см/пиксель исходного растра; не утверждать физическую точность без калибровки.
7. Не хранить .venv/, веса, output/, demo_jobs/, пользовательские примеры, модели, датасет и runtime-логи в Git.
8. Не менять лицензионную стратегию и не заявлять коммерческую пригодность сторонних кодовых/датасетных/весовых артефактов без проверки прав.
9. При изменениях модели не смешивать несвязанные переделки geometry, UI и serialization. Сначала локализовать сбой на промежуточных артефактах.

## Рабочий цикл

1. Установить тип проблемы: маска/ROI, классификация проёмов, топология помещений, геометрия, привязки, quality gate или API.
2. Зафиксировать вход и воспроизводимую команду; проверить исходное изображение, overlay и промежуточные JSON.
3. Добавить минимальный регрессионный тест для конкретной причины; избегать эвристики, улучшающей лишь один пример.
4. Запустить тесты затронутого модуля и минимум clean-baseline обоих конвейеров.
5. При изменении распознавания/геометрии прогнать весь манифест сложных случаев; не приписывать accuracy результатам без ground truth.
6. Проверить результаты gate, экспортные поля, bindings и наличие/отсутствие ссылок для invalid в demo.
7. Обновить [CURRENT_STATUS.md](docs/CURRENT_STATUS.md), соответствующий технический документ и roadmap, если поменялись подтверждённые возможности или приоритеты.

Команды (требуют настроенной среды, весов и для регрессии локальных exmpls/):

~~~bash
python -m unittest discover -s test -p "test_*.py"
node --test test/demo_quality_ui.test.cjs
python scripts/run_recognition_regression.py --case clean-baseline
python scripts/run_recognition_regression.py --case clean-baseline --pipeline-v2
python scripts/run_recognition_regression.py --pipeline-v2
~~~

## Отбор новых весов

- Разделять train/val/test на уровне семейств исходников, исключать дубликаты/утечки.
- Новые растры могут иметь частичную разметку с ignored pixels (255): метрики считать только по размеченным пикселям.
- Раздельно сравнивать structural SVG, full-scene SVG и новые растры; полнота дверей/окон важнее незначительного улучшения среднего IoU.
- Хранить baseline, seed, manifest, хеш весов, конфигурацию и отчёты кандидата отдельно.
- Не использовать test split для подбора; не заменять production/serving weights автоматически.
- После допуска кандидат обязан пройти полный CPU pipeline, quality gate и контрольный эталон.

Первый смешанный пилот **отклонён**; full-scene pilot **подготовлен, не обучен** (срез 12.11.2026). Фактические цифры и протокол: [docs/CURRENT_STATUS.md](docs/CURRENT_STATUS.md), [docs/TRAINING_PILOT.md](docs/TRAINING_PILOT.md), [docs/FULL_SCENE_TRAINING.md](docs/FULL_SCENE_TRAINING.md).

## Definition of Done

Изменение не считается завершённым, пока воспроизводимые тесты не проходят, исходная совместимость не нарушена, проблемы на сложных планах не маскируются, invalid-экспорт заблокирован и документация отражает **измеренное**, а не предполагаемое улучшение.
