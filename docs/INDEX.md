# Навигация по документации

Обновлено 12 ноября 2026 года. При расхождении старых результатов и актуальной реализации начните с [CURRENT_STATUS.md](CURRENT_STATUS.md) и кода в ветке main.

## Начать здесь

| Документ | Для чего нужен |
| --- | --- |
| [README в корне](../README.md) | Назначение, зависимости, запуск CLI и demo |
| [CURRENT_STATUS.md](CURRENT_STATUS.md) | Что реализовано, что экспериментально, последние измерения и приоритеты |
| [DEVELOPMENT_MAP.md](DEVELOPMENT_MAP.md) | Границы модулей и форматы данных |
| [DEMO_API.md](DEMO_API.md) | HTTP-маршруты, обработка заданий, выбор пользователем и запреты экспорта |
| [AGENTS.md](../AGENTS.md) | Контракты и правила для Codex/разработчиков |
| [RECOGNITION_ROADMAP.md](RECOGNITION_ROADMAP.md) | Актуальный порядок следующих работ |

## Конвейер распознавания

| Документ | Содержание |
| --- | --- |
| [PIPELINE_V2.md](PIPELINE_V2.md) | ROI, dual-pass и восстановление топологии комнат |
| [OPENING_CLASSIFICATION.md](OPENING_CLASSIFICATION.md) | Эвристики и подтверждения дверей/окон |
| [OPENING_RECOVERY.md](OPENING_RECOVERY.md) | Восстановление пропущенных проёмов |
| [TOPOLOGY_WINDOW_RECOVERY.md](TOPOLOGY_WINDOW_RECOVERY.md) | Восстановление окон по топологии |
| [POLYGON_GEOMETRY.md](POLYGON_GEOMETRY.md) | Экспорт сложных многоугольников и общих границ |
| [INTERNAL_WALL_GRAPH.md](INTERNAL_WALL_GRAPH.md) | Осевые линии перегородок, T/X-узлы |
| [QUALITY_GATE.md](QUALITY_GATE.md) | Структурная проверка и блокировка некорректных экспортов |
| [USER_REVIEW.md](USER_REVIEW.md) | Решения пользователя по неоднозначным объектам |
| [STRUCTURAL_RECOGNITION_PATTERNS.md](STRUCTURAL_RECOGNITION_PATTERNS.md) | Известные структурные паттерны распознавания |

## Регрессия, разметка и обучение

| Документ | Содержание |
| --- | --- |
| [REGRESSION_BASELINE.md](REGRESSION_BASELINE.md) | Исторический CPU baseline и повторяемые проверки |
| [GROUND_TRUTH.md](GROUND_TRUTH.md) | Разметка эталонных результатов |
| [RASTER_ANNOTATIONS.md](RASTER_ANNOTATIONS.md) | Частичные растровые аннотации и ignored pixels |
| [TRAINING_PLAN.md](TRAINING_PLAN.md) | Данные, критерии допусков, журнал экспериментов |
| [TRAINING_PREFLIGHT.md](TRAINING_PREFLIGHT.md) | Аудит датасета, проверка стартовых весов |
| [TRAINING_PILOT.md](TRAINING_PILOT.md) | Первый смешанный пилот и причины отказа |
| [FULL_SCENE_TRAINING.md](FULL_SCENE_TRAINING.md) | Полная сцена SVG: мебель, подписи, двойной контроль качества |
| [data/README.md](../data/README.md) | Получение CubiCasa5K и организация данных |
| [hf_model_card.md](hf_model_card.md) | Карточка исходной модели |

**Как трактовать документы:** числовые результаты относятся к конкретным датам, входным изображениям, версиям весов и настройкам. Исторические таблицы не гарантируют такого же качества после изменений. Новая модель не считается одобренной, пока не прошла зафиксированный протокол оценки, полный конвейер и проверку экспорта.
