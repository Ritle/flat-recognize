# flat-recognize

Экспериментальный CPU-ориентированный сервис распознавания растровых планировок квартир и экспорта редактируемого проекта в JSON целевого приложения.

**Вход:** PNG, JPG/JPEG, WEBP → **распознавание:** стены, помещения, двери, окна → **выход:** project.json (connectors, segments, areas, elements). Это не готовый автоматический конвертер для любых планов: на сложных изображениях результат может требовать проверки или блокироваться.

**Актуальное состояние:** 12 ноября 2026 года. [Текущие возможности, ограничения и измерения](docs/CURRENT_STATUS.md) · [Карта документации](docs/INDEX.md).

## Два конвейера

| | V1 | V2 |
| --- | --- | --- |
| Статус | По умолчанию | Экспериментальный, включается явно |
| Подготовка и inference | Сегментация исходного растра | ROI/нормализация и консервативное объединение двух проходов |
| Помещения | Базовое извлечение по барьерной маске | Восстановление топологии, дополнительные проверки окон и перегородок |
| Геометрия экспорта | Bbox | Полигональные границы, общие узлы, внутренние стены |
| Экспорт | Проверяется quality gate | Проверяется тем же quality gate |

V2 не гарантирует лучший результат для каждого изображения. Оба режима сохраняют исходные координаты растра и проверяют валидность геометрии и привязок. Решения пользователя не отключают проверку качества.

## Быстрый старт: растровый CLI и demo

Требуется Python 3.11 (в репозитории зафиксирован 3.11.9), рабочая установка PyTorch, зависимости serve и веса модели. Для CPU на Windows используйте проверенное окружение и версии из [REGRESSION_BASELINE.md](docs/REGRESSION_BASELINE.md).

~~~powershell
git clone https://github.com/Ritle/flat-recognize.git
cd flat-recognize
py -3.11 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -c configs/constraints-cpu.txt torch torchvision --index-url https://download.pytorch.org/whl/cpu
& .\.venv\Scripts\python.exe -m pip install -c configs/constraints-cpu.txt -e ".[serve]"
New-Item -ItemType Directory -Path weights -Force
curl.exe -fL -o weights/best.safetensors https://huggingface.co/Yytsi/floorplan-to-3d-walls/resolve/main/best.safetensors
curl.exe -fL -o weights/config.yaml https://huggingface.co/Yytsi/floorplan-to-3d-walls/resolve/main/config.yaml

# Запуск на собственном JPG/PNG/WEBP (подставьте существующий файл):
& .\.venv\Scripts\python.exe scripts/process_floorplan.py "path/to/plan.png" --pipeline-v2

# Локальный веб-интерфейс:
& .\.venv\Scripts\python.exe -m uvicorn demo.app:app --host 127.0.0.1 --port 8081
~~~

Адрес demo: http://127.0.0.1:8081. В интерфейсе можно выбрать экспериментальный V2, просмотреть диагностику, принять решение по предложенным спорным объектам и скачать разрешённый к экспорту JSON.

На Linux/macOS создайте и активируйте venv обычным способом, установите совместимую сборку PyTorch и зависимости serve, затем запустите:

~~~bash
python scripts/process_floorplan.py path/to/plan.png --pipeline-v2
python -m uvicorn demo.app:app --host 127.0.0.1 --port 8081
~~~

Веса и входные изображения не включены в Git. Для растрового CLI нативный Cairo не нужен; он понадобится для SVG-рендеринга, SVG-обучения и исходного 3D-примера.

## Запуск, проверка и результаты

~~~bash
# V1 (режим по умолчанию)
python scripts/process_floorplan.py path/to/plan.png

# V2
python scripts/process_floorplan.py path/to/plan.png --pipeline-v2

# Явный путь вывода и масштаб (сантиметров на пиксель)
python scripts/process_floorplan.py path/to/plan.png --pipeline-v2 --scale 1.0 --output output/project.json

# CPU регрессия по эталону / всему манифесту
python scripts/run_recognition_regression.py --case clean-baseline --pipeline-v2
python scripts/run_recognition_regression.py --pipeline-v2

# Юнит-тесты и тесты интерфейса
python -m unittest discover -s test -p "test_*.py"
node --test test/demo_quality_ui.test.cjs
~~~

Регрессия использует 19 кейсов из test/recognition_cases.json. Сами изображения из локального каталога exmpls/ не входят в Git: без них соответствующие прогоны не воспроизводятся. Основные выходные артефакты сохраняются в output/, результаты demo — в demo_jobs/ (оба каталога не версионируются).

**Quality gate** использует статусы good, review, invalid. good/review допускают скачивание с соответствующими предупреждениями; invalid блокирует API-загрузку проекта. Это проверка структурной согласованности, **не** оценка точности относительно истинной разметки. Описание API — [DEMO_API.md](docs/DEMO_API.md), контракт — [QUALITY_GATE.md](docs/QUALITY_GATE.md).

## Архитектура

~~~text
растр → predict_raster → classify_openings
      → extract_rooms (V1) / цепочка topology-repair (V2)
      → build_project_json → add_openings → add_windows
      → validate_recognition → project.json
~~~

Ключевые каталоги:

- scripts/ — оркестрация, постобработка, адаптер JSON, проверки и регрессия.
- src/buildingcv/ — модель, инференс, подготовка данных, обучение и оценка.
- demo/ — отдельный FastAPI интерфейс для растрового сценария.
- test/ — юнит-тесты, манифест регрессии и шаблоны целевого JSON.
- docs/ — архитектура, измерения, ограничения и экспериментальные протоколы.
- viewer/ и server/ — унаследованный SVG→3D-пример; это **не** растровый demo.

## Дообучение

Ветки экспериментов не заменяют рабочие веса автоматически. Первый смешанный пилот с частичными растровыми масками не прошёл критерии улучшения дверей/окон; эксперимент на полных SVG-сценах подготовлен, но на зафиксированном состоянии ещё не обучен. Подробнее: [CURRENT_STATUS.md](docs/CURRENT_STATUS.md), [TRAINING_PLAN.md](docs/TRAINING_PLAN.md) и [FULL_SCENE_TRAINING.md](docs/FULL_SCENE_TRAINING.md).

## Происхождение и лицензии

Исходная модель и код SVG→3D базируются на [Yytsi/floorplan-to-3d](https://github.com/Yytsi/floorplan-to-3d) и данных [CubiCasa5K](https://github.com/CubiCasa/CubiCasa5k). Перед коммерческим применением необходимо отдельно проверить права на код, датасеты, разметку и скачанные веса: наличие работающего PoC не является разрешением на использование сторонних материалов.

Исходный SVG-пример при наличии весов запускается через ./dev.sh (macOS/Linux) или python -m uvicorn server.main:app --reload --host 127.0.0.1 --port 8000. Его сервер и API /extract не заменяют растровый API /api/recognize.
