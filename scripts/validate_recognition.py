"""Deterministic export checks. The quality grade is not model confidence/accuracy."""

import argparse
import json
import math
from pathlib import Path

from shapely.geometry import Polygon


def finite(value):
    try:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    except OverflowError:
        return False


def validate_recognition(rooms, project=None, bindings=None, scale=1.0, geometry=None):
    issues = []
    metrics = {}

    def issue(code, severity, message, **context):
        issues.append(dict(code=code, severity=severity, message=message, **context))

    if geometry is not None:
        if not isinstance(geometry, dict):
            issue("invalid_geometry_report", "error", "Некорректный отчёт геометрии экспорта.")
            geometry = {}
        metrics["geometry_adapter"] = geometry.get("adapter")
        for message in geometry.get("errors", []):
            issue("unsupported_geometry", "error", message)
        for message in geometry.get("warnings", []):
            issue("geometry_warning", "warning", message)

    def objects(data, key, location):
        value = data.get(key)
        if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
            issue("invalid_array", "error", f"Некорректный список {location}.{key}.")
            return []
        return value

    def polygon(points, location):
        if (not isinstance(points, list) or len(points) < 3
                or any(not isinstance(p, (list, tuple)) or len(p) != 2
                       or not all(finite(v) for v in p) for p in points)):
            issue("invalid_polygon", "error", f"Некорректные координаты: {location}.")
            return None
        result = Polygon(points)
        if not result.is_valid or result.area <= 1e-6:
            issue("invalid_polygon", "error", f"Контур вырожден или пересекает себя: {location}.")
            return None
        return result

    def overlaps(polygons, location):
        maximum = 0.0
        for i, (name, a) in enumerate(polygons):
            for other, b in polygons[i + 1:]:
                ratio = a.intersection(b).area / min(a.area, b.area)
                maximum = max(maximum, ratio)
                if ratio >= 0.2:
                    issue("massive_overlap", "error", f"Значительное наложение помещений: {name}, {other}.",
                          overlap_ratio=round(ratio, 4))
                elif ratio >= 0.05:
                    issue("room_overlap", "warning", f"Проверьте наложение помещений: {name}, {other}.",
                          overlap_ratio=round(ratio, 4))
        metrics[location + "_max_overlap_ratio"] = round(maximum, 4)

    if not isinstance(rooms, dict):
        issue("invalid_rooms", "error", "Результат выделения комнат должен быть JSON-объектом.")
        rooms = {}
    source_rooms = objects(rooms, "rooms", "recognition")
    if not source_rooms:
        issue("no_rooms", "error", "Замкнутые помещения не найдены; экспорт заблокирован.")
    source_polygons = []
    for index, room in enumerate(source_rooms):
        name = f"исходная комната {index + 1}"
        shape = polygon(room.get("polygon"), name)
        if shape is not None:
            source_polygons.append((name, shape))
        if room.get("uncertain"):
            issue("uncertain_room", "warning", f"{name}: граница восстановлена предположительно.")
    overlaps(source_polygons, "source")
    diagnostics = rooms.get("diagnostics", {})
    if not isinstance(diagnostics, dict):
        diagnostics = {}
    diagnostic_warnings = diagnostics.get("warnings", [])
    for warning in diagnostic_warnings if isinstance(diagnostic_warnings, list) else []:
        if geometry and geometry.get("adapter") == "polygon" and warning == "Есть помещения сложной формы; текущий экспорт упрощает их до прямоугольников.":
            continue
        if isinstance(warning, str):
            issue("room_diagnostic", "warning", warning)
    outside = diagnostics.get("outside_fraction_in_structure")
    if finite(outside):
        metrics["outside_fraction_in_structure"] = outside
        if outside >= 0.25:
            issue("possible_leak", "warning", "Внешний фон занимает значительную часть области плана; возможна потеря комнат.")
    doors = objects(rooms, "doors", "recognition")
    windows = objects(rooms, "windows", "recognition")
    if not windows:
        issue("no_windows", "warning", "Окна не распознаны; проверьте проёмы на исходном изображении.")
    if not finite(scale) or scale <= 0:
        issue("invalid_scale", "error", "Масштаб должен быть положительным конечным числом.")
    else:
        issue("approximate_scale", "info", f"Масштаб {scale:g} см/пиксель задан без проверки по известной длине; физические размеры приблизительны.")
    metrics.update(rooms=len(source_rooms), detected_doors=len(doors), detected_windows=len(windows))

    exported = {"doorItem": 0, "windowItem": 0}
    all_segments = {}
    if project is not None:
        if not isinstance(project, dict):
            issue("invalid_project", "error", "Проект должен быть JSON-объектом.")
            project = {}
        levels = objects(project, "levels", "project")
        if not levels:
            issue("no_levels", "error", "В проекте нет этажей.")
        area_count = 0
        segment_count = connector_count = 0
        for index, level in enumerate(levels):
            label = f"этаж {index + 1}"

            def index_objects(key):
                result = {}
                for item in objects(level, key, label):
                    uid = item.get("uuid")
                    if not isinstance(uid, str) or not uid or uid in result:
                        issue("invalid_uuid", "error", f"{label}: отсутствующий или повторный UUID в {key}.")
                    else:
                        result[uid] = item
                return result

            connectors = index_objects("connectors")
            segments = index_objects("segments")
            areas = index_objects("areas")
            elements = index_objects("elements")
            segment_count += len(segments)
            connector_count += len(connectors)
            coords = {}
            for uid, connector in connectors.items():
                x, y = connector.get("x"), connector.get("y")
                if not finite(x) or not finite(y):
                    issue("invalid_coordinate", "error", f"{label}: некорректные координаты узла {uid}.")
                else:
                    coords[uid] = (x, y)
            edges = set()
            neighbours = {uid: set() for uid in connectors}
            for uid, segment in segments.items():
                start, end = segment.get("start"), segment.get("end")
                if not isinstance(start, str) or not isinstance(end, str) or start not in coords or end not in coords:
                    issue("missing_connector", "error", f"{label}: стена {uid} ссылается на отсутствующий узел.")
                    continue
                length = math.dist(coords[start], coords[end])
                if length <= 1e-6:
                    issue("zero_segment", "error", f"{label}: стена {uid} имеет нулевую длину.")
                edges.add(frozenset((start, end)))
                neighbours[start].add(end)
                neighbours[end].add(start)
                all_segments[uid] = segment
                thickness = segment.get("thickness")
                height = segment.get("height")
                if not finite(thickness) or thickness <= 0 or not finite(height) or height <= 0:
                    issue("invalid_wall_size", "error", f"{label}: некорректная толщина или высота стены {uid}.")
            for uid, connector in connectors.items():
                adjacency = connector.get("adjacency")
                if (not isinstance(adjacency, list) or any(not isinstance(c, str) for c in adjacency)
                        or set(adjacency) != neighbours[uid]):
                    issue("invalid_adjacency", "error", f"{label}: связи узла {uid} не соответствуют стенам.")
            shapes = []
            for uid, area in areas.items():
                cycle = area.get("connectors")
                if (not isinstance(cycle, list) or len(cycle) < 3
                        or any(not isinstance(c, str) or c not in coords for c in cycle)):
                    issue("invalid_area", "error", f"{label}: некорректный контур помещения {uid}.")
                    continue
                # Both implicit and explicit closure are accepted.
                if cycle[-1] == cycle[0]:
                    cycle = cycle[:-1]
                if len(set(cycle)) != len(cycle):
                    issue("invalid_area", "error", f"{label}: повторные узлы контура {uid}.")
                if any(frozenset((a, b)) not in edges for a, b in zip(cycle, cycle[1:] + cycle[:1])):
                    issue("unclosed_area", "error", f"{label}: граница помещения {uid} не замкнута стенами.")
                shape = polygon([coords[c] for c in cycle], f"{label}, помещение {uid}")
                if shape is not None:
                    shapes.append((uid, shape))
            overlaps(shapes, f"level_{index}")
            area_count += len(areas)
            for uid, element in elements.items():
                kind = element.get("type")
                if kind not in exported:
                    continue
                exported[kind] += 1
                wall = element.get("wall")
                if not isinstance(wall, dict):
                    wall = {}
                wall_id = wall.get("uuid")
                if not isinstance(wall_id, str) or wall_id not in segments:
                    issue("missing_wall", "error", f"Проём {uid} ссылается на отсутствующую стену.")
                position = wall.get("position")
                if not finite(position) or not 0 <= position <= 100:
                    issue("opening_position", "error", f"Положение проёма {uid} выходит за пределы стены.")
                size = element.get("size", {})
                if (not isinstance(size, dict) or any(not finite(size.get(k)) or size[k] <= 0 for k in ("x", "y"))):
                    issue("opening_size", "error", f"Некорректные размеры проёма {uid}.")
                elif isinstance(wall_id, str) and wall_id in segments and finite(position):
                    segment = segments[wall_id]
                    a, b = segment.get("start"), segment.get("end")
                    if isinstance(a, str) and isinstance(b, str) and a in coords and b in coords:
                        length = math.dist(coords[a], coords[b])
                        half = size["x"] / 2
                        center = length * position / 100
                        if center - half < -5 or center + half > length + 5:
                            issue("opening_exceeds_segment", "warning", f"Проём {uid} выходит за конец стены; проверьте ширину и привязку.")
                materials = element.get("materials", [])
                if not isinstance(materials, list) or any(not isinstance(m, dict) or m.get("itemUuid") != uid for m in materials):
                    issue("material_reference", "error", f"Материалы проёма {uid} ссылаются на другой объект.")
                if kind == "doorItem" and not element.get("model"):
                    issue("no_door_leaf", "error", f"У двери {uid} отсутствует модель дверного полотна.")
        if area_count != len(source_rooms):
            issue("room_count_mismatch", "error", "Число помещений в проекте отличается от результата распознавания.")
        for kind, detected in (("doorItem", doors), ("windowItem", windows)):
            if exported[kind] != len(detected):
                issue("unbound_openings", "warning", f"Не все проёмы перенесены в проект ({kind}: {exported[kind]} из {len(detected)}).")
        metrics.update(exported_rooms=area_count, exported_doors=exported["doorItem"], exported_windows=exported["windowItem"],
                       exported_segments=segment_count, exported_connectors=connector_count)
        if bindings is None:
            issue("missing_binding_diagnostics", "warning", "Нет диагностики расстояния проёмов до стен.")
        else:
            if not isinstance(bindings, dict):
                issue("invalid_binding", "error", "Диагностика привязок должна быть JSON-объектом.")
                bindings = {}
            for key in ("doors", "windows"):
                rows = objects(bindings, key, "bindings")
                if len(rows) != exported["doorItem" if key == "doors" else "windowItem"]:
                    issue("binding_count_mismatch", "error", "Число диагностических привязок отличается от числа проёмов проекта.")
                for binding in rows:
                    distance = binding.get("distance_to_wall_cm")
                    wall_id = binding.get("wall_uuid")
                    segment = all_segments.get(wall_id) if isinstance(wall_id, str) else None
                    if segment is None or not finite(distance) or distance < 0:
                        issue("invalid_binding", "error", "Некорректная диагностическая привязка проёма.")
                        continue
                    thickness = segment.get("thickness", 20)
                    threshold = max(20, thickness * 1.5) if finite(thickness) else 30
                    if distance > threshold:
                        issue("opening_far_from_wall", "warning", f"Проём {key} находится далеко от стены ({distance:g} см).",
                              distance_cm=distance, threshold_cm=threshold)
    errors = list(dict.fromkeys(i["message"] for i in issues if i["severity"] == "error"))
    warnings = list(dict.fromkeys(i["message"] for i in issues if i["severity"] != "error"))
    warning_codes = {i["code"] for i in issues if i["severity"] == "warning"}
    status = "invalid" if errors else "review" if warning_codes else "good"
    return dict(version=1, status=status, export_allowed=not errors,
                quality=0.0 if errors else round(max(0.1, 1 - len(warning_codes) * 0.08), 2),
                warnings=warnings, errors=errors, issues=issues, metrics=metrics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rooms", type=Path, required=True)
    parser.add_argument("--project", type=Path)
    parser.add_argument("--bindings", type=Path)
    parser.add_argument("--geometry", type=Path)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    def load(path):
        return json.loads(path.read_text(encoding="utf-8")) if path else None

    try:
        result = validate_recognition(load(args.rooms), load(args.project), load(args.bindings), args.scale, load(args.geometry))
    except (OSError, ValueError) as exc:
        result = dict(version=1, status="invalid", quality=0.0, export_allowed=False,
                      warnings=[], errors=[f"Не удалось прочитать диагностические JSON: {exc}"], issues=[], metrics={})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(f"QUALITY: {result['status']} (grade={result['quality']})")
    for message in result["warnings"] + result["errors"]:
        print(message)
    return 0 if result["export_allowed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
