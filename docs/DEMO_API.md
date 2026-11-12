# Растровое demo и API

Актуально для demo/app.py на 12 ноября 2026 года. FastAPI-приложение demo.app:app — основной ручной интерфейс распознавания PNG/JPG/WEBP, **не** server.main:app (тот обслуживает унаследованный SVG→3D viewer).

## Локальный запуск

~~~powershell
& .\.venv\Scripts\python.exe -m uvicorn demo.app:app --host 127.0.0.1 --port 8081
~~~

Откройте http://127.0.0.1:8081. Установите зависимости serve и положите совместимые best.safetensors и config.yaml в weights/ до распознавания.

## Маршруты

| Метод | Путь | Назначение |
| --- | --- | --- |
| GET | / | HTML интерфейс demo |
| POST | /api/recognize | Загрузка растра и синхронный запуск pipeline |
| GET | /api/jobs/{job_id}/quality | Подробный отчёт проверки качества |
| GET | /api/jobs/{job_id}/opening-preflight | Проверки привязки проёмов, если отчёт сформирован |
| GET | /api/jobs/{job_id}/json | JSON проекта, **только** при разрешённом экспорте |
| GET | /api/jobs/{job_id}/download | Скачивание project.json, **только** при разрешённом экспорте |
| GET | /api/jobs/{job_id}/image/{kind} | Исходное изображение или доступный диагностический слой |

POST /api/recognize принимает multipart/form-data:

- file — обязательный файл с расширением jpg, jpeg, png или webp (до 20 МБ).
- pipeline_v2 — необязательное булево поле, по умолчанию false.
- decisions — необязательная JSON-строка с решениями пользователя по кандидатам.

~~~bash
curl -F "file=@path/to/plan.png" -F "pipeline_v2=true" http://127.0.0.1:8081/api/recognize
~~~

Поля ответа включают ok, job_id, filename, seconds, stats, images, pipeline, warnings, quality, inference, review_questions, geometry_review_candidates, possible_room_partitions, opening_preflight, exterior_enclosures, applied_decisions, image_sha256, original_url, quality_url, opening_preflight_url, download_url, json_url.

Если quality.status = invalid, ответ распознавания может содержать диагностику, но ok=false, download_url/json_url=null. Повторный GET экспортного endpoint возвращает 409, а не некорректный проект. Для good и review скачивание доступно; review несёт предупреждения.

## Подтверждение спорных объектов

Клиент может повторно послать **тот же исходный файл** вместе с decisions. Базовая структура:

~~~json
{
  "version": 1,
  "image_sha256": "SHA-256 исходных байтов",
  "answers": [
    {"opening_id": 8, "fingerprint": "отпечаток кандидата", "choice": "ignore"}
  ]
}
~~~

Ответы адресуются ровно одному объекту (opening_id, room_id, partition_id или exterior_id), содержат fingerprint и допустимый для типа объекта choice. Вопросы и fingerprints следует брать **из ответа API**, а не конструировать вручную. Для дверей/окон предусмотрены door, window, ignore; для внешнего контура — confirm, ignore. Вопросы про перегородки и внешний контур доступны только в V2. Данные привязаны к SHA-256 исходного изображения и кандидату; устаревшие/несовпадающие ответы отклоняются.

При выборе комнат, проёмов и границ необходимые этапы пересчитываются, после чего снова выполняется структурная проверка. Подтверждение кандидата внешнего контура **не** является безусловным созданием комнаты: гипотеза проходит отдельную проверку сохранения существующих помещений. Решения не отключают gate и не гарантируют статус good.

Подробности: [USER_REVIEW.md](USER_REVIEW.md), [QUALITY_GATE.md](QUALITY_GATE.md), scripts/review_exterior_enclosures.py.

## Диагностика, хранение, ограничения

Каждому запросу присваивается UUID job_id; файлы сохраняются локально в demo_jobs/{job_id}/ и output/. JSON отчёты и overlays полезны для поиска ошибочной стадии. Возможные виды изображения включают original, segmentation, classified, rooms, barrier, export, wall_geometry, exterior_enclosures, exterior_trial и специфичные для V2 preprocessing/normalized/segmentation_original/segmentation_normalized/segmentation_fused/closures/recovery/topology_windows. Не все слои существуют в каждом прогоне.

Это локальная разработческая demo: API не следует выставлять в интернет без отдельной работы по аутентификации, изоляции запуска, ограничениям ресурсов и очистке заданий. Нет гарантий многопользовательской изоляции и production SLA.
