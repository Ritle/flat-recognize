import argparse
import copy
import json
import math
import uuid
from pathlib import Path


def polygon_area(points):
    s = 0.0
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def snap_value(v, values, tolerance):
    for existing in values:
        if abs(existing - v) <= tolerance:
            return existing
    values.append(v)
    return v


def connector_num(cid):
    if cid.startswith("c"):
        try:
            return int(cid[1:])
        except ValueError:
            pass
    return 0


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("rooms_json")
    parser.add_argument("template_json")

    parser.add_argument(
        "--output",
        default="output/project.json"
    )

    # Для текущей картинки 1 px ≈ 1 cm выглядит правдоподобно.
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="centimeters per pixel"
    )

    parser.add_argument(
        "--wall-thickness",
        type=float,
        default=20.0
    )

    parser.add_argument(
        "--height",
        type=float,
        default=270.0
    )

    parser.add_argument(
        "--snap",
        type=float,
        default=8.0,
        help="coordinate snapping tolerance in px"
    )
    parser.add_argument("--geometry-mode", choices=("bbox", "auto", "polygon"), default="bbox")
    parser.add_argument("--geometry-report", help="Separate export geometry diagnostics JSON")
    parser.add_argument("--source-image", help="Original raster for export overlay")
    parser.add_argument("--overlay", help="Export geometry overlay PNG")

    args = parser.parse_args()

    rooms_path = Path(args.rooms_json)
    template_path = Path(args.template_json)

    with rooms_path.open("r", encoding="utf-8") as f:
        src = json.load(f)

    with template_path.open("r", encoding="utf-8") as f:
        template = json.load(f)

    rooms = src.get("rooms", [])

    if not rooms:
        raise RuntimeError("No rooms in source JSON")

    if not template.get("levels"):
        raise RuntimeError("Template has no levels")

    template_level = template["levels"][0]
    geometry_errors = []
    if args.geometry_mode != "bbox":
        from polygon_geometry import build_polygon_project, requires_polygons, render_geometry
        from shapely.errors import GEOSException
        if args.geometry_mode == "polygon" or requires_polygons(rooms):
            try:
                result, report = build_polygon_project(src, template, args.scale, args.wall_thickness, args.height)
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
                report_path = Path(args.geometry_report) if args.geometry_report else output_path.with_suffix(".geometry.json")
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
                if args.overlay and args.source_image:
                    render_geometry(result, report, args.source_image, args.overlay)
                print(f"Polygon geometry created: {output_path}; unsupported boundaries: {len(report['unsupported_segments'])}")
                return
            except (ValueError, GEOSException) as exc:
                # Preserve a diagnostic candidate, but explicitly block its export.
                geometry_errors.append(f"Полигональная геометрия не построена: {exc}")

    # -------------------------------------------------
    # Сохраняем дефолтные настройки сегмента/помещения
    # из реального файла проекта.
    # -------------------------------------------------

    segment_template = {}

    if template_level.get("segments"):
        segment_template = copy.deepcopy(
            template_level["segments"][0]
        )

    area_template = {}

    if template_level.get("areas"):
        area_template = copy.deepcopy(
            template_level["areas"][0]
        )

    # -------------------------------------------------
    # Определяем центр геометрии template.
    #
    # Новую квартиру помещаем примерно туда же,
    # чтобы существующая camera с большей вероятностью
    # сразу её показывала.
    # -------------------------------------------------

    old_connectors = template_level.get(
        "connectors",
        []
    )

    if old_connectors:
        old_x = [float(c["x"]) for c in old_connectors]
        old_y = [float(c["y"]) for c in old_connectors]

        template_center_x = (
            min(old_x) + max(old_x)
        ) / 2.0

        template_center_y = (
            min(old_y) + max(old_y)
        ) / 2.0
    else:
        template_center_x = 0.0
        template_center_y = 0.0

    # -------------------------------------------------
    # Строим осевые прямоугольники комнат.
    #
    # bbox комнаты находится по внутренней стороне стены.
    # Поэтому сдвигаем каждую сторону наружу примерно
    # на половину толщины стены.
    # -------------------------------------------------

    half_wall_px = (
        args.wall_thickness /
        args.scale /
        2.0
    )

    room_rects = []

    for idx, room in enumerate(rooms):
        bbox = room.get("bbox")

        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = map(float, bbox)
        else:
            pts = room.get("polygon", [])

            if not pts:
                continue

            xs = [float(p[0]) for p in pts]
            ys = [float(p[1]) for p in pts]

            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)

        room_rects.append({
            "room_id": idx,
            "x1": x1 - half_wall_px,
            "y1": y1 - half_wall_px,
            "x2": x2 + half_wall_px,
            "y2": y2 + half_wall_px,
        })

    # -------------------------------------------------
    # Snap близких параллельных стен.
    #
    # У двух соседних помещений осевые линии могут
    # отличаться на несколько пикселей.
    # -------------------------------------------------

    snap_x = []
    snap_y = []

    for r in room_rects:
        r["x1"] = snap_value(
            r["x1"],
            snap_x,
            args.snap
        )
        r["x2"] = snap_value(
            r["x2"],
            snap_x,
            args.snap
        )
        r["y1"] = snap_value(
            r["y1"],
            snap_y,
            args.snap
        )
        r["y2"] = snap_value(
            r["y2"],
            snap_y,
            args.snap
        )

    # -------------------------------------------------
    # Все границы помещений как линии.
    # -------------------------------------------------

    raw_edges = []

    for r in room_rects:
        rid = r["room_id"]

        x1 = r["x1"]
        x2 = r["x2"]
        y1 = r["y1"]
        y2 = r["y2"]

        raw_edges.extend([
            {
                "orientation": "h",
                "coord": y1,
                "a": x1,
                "b": x2,
                "room": rid,
                "side": "top",
            },
            {
                "orientation": "v",
                "coord": x2,
                "a": y1,
                "b": y2,
                "room": rid,
                "side": "right",
            },
            {
                "orientation": "h",
                "coord": y2,
                "a": x1,
                "b": x2,
                "room": rid,
                "side": "bottom",
            },
            {
                "orientation": "v",
                "coord": x1,
                "a": y1,
                "b": y2,
                "room": rid,
                "side": "left",
            },
        ])

    # -------------------------------------------------
    # Группируем линии по одной координате.
    # Потом делим их во всех точках начала/конца.
    #
    # Это автоматически создаёт корректные T-junction.
    # -------------------------------------------------

    line_groups = {}

    for edge in raw_edges:
        key = (
            edge["orientation"],
            round(edge["coord"], 4)
        )

        line_groups.setdefault(
            key,
            []
        ).append(edge)

    atomic_edges = []

    for (orientation, coord), edges in line_groups.items():

        cuts = sorted({
            round(e["a"], 4)
            for e in edges
        } | {
            round(e["b"], 4)
            for e in edges
        })

        for i in range(len(cuts) - 1):
            a = cuts[i]
            b = cuts[i + 1]

            if b - a < 1:
                continue

            mid = (a + b) / 2.0

            owners = []

            for e in edges:
                lo = min(e["a"], e["b"])
                hi = max(e["a"], e["b"])

                if lo - 0.01 <= mid <= hi + 0.01:
                    owners.append(e)

            if not owners:
                continue

            atomic_edges.append({
                "orientation": orientation,
                "coord": coord,
                "a": a,
                "b": b,
                "owners": owners,
            })

    # -------------------------------------------------
    # Убираем дубли atomic edge.
    # -------------------------------------------------

    unique_atomic = {}

    for e in atomic_edges:
        key = (
            e["orientation"],
            round(e["coord"], 4),
            round(e["a"], 4),
            round(e["b"], 4),
        )

        if key not in unique_atomic:
            unique_atomic[key] = e
        else:
            unique_atomic[key]["owners"].extend(
                e["owners"]
            )

    atomic_edges = list(
        unique_atomic.values()
    )

    # -------------------------------------------------
    # Координатная система.
    # -------------------------------------------------

    all_px_points = []

    for e in atomic_edges:
        if e["orientation"] == "h":
            all_px_points.extend([
                (e["a"], e["coord"]),
                (e["b"], e["coord"]),
            ])
        else:
            all_px_points.extend([
                (e["coord"], e["a"]),
                (e["coord"], e["b"]),
            ])

    min_px_x = min(p[0] for p in all_px_points)
    max_px_x = max(p[0] for p in all_px_points)

    min_px_y = min(p[1] for p in all_px_points)
    max_px_y = max(p[1] for p in all_px_points)

    new_width = (
        max_px_x - min_px_x
    ) * args.scale

    new_height = (
        max_px_y - min_px_y
    ) * args.scale

    origin_x = (
        template_center_x -
        new_width / 2.0
    )

    origin_y = (
        template_center_y -
        new_height / 2.0
    )

    def to_world(x, y):
        return (
            origin_x +
            (x - min_px_x) * args.scale,

            origin_y +
            (y - min_px_y) * args.scale,
        )

    # -------------------------------------------------
    # Connectors
    # -------------------------------------------------

    connector_by_coord = {}
    connectors = []

    def get_connector(x_px, y_px):
        x, y = to_world(x_px, y_px)

        key = (
            round(x, 4),
            round(y, 4),
        )

        if key in connector_by_coord:
            return connector_by_coord[key]

        cid = f"c{len(connectors)}"

        c = {
            "uuid": cid,
            "x": round(x, 4),
            "y": round(y, 4),
            "z": 0,
            "adjacency": [],
        }

        connectors.append(c)
        connector_by_coord[key] = cid

        return cid

    # -------------------------------------------------
    # Segments
    # -------------------------------------------------

    segments = []

    segment_lookup = {}

    for edge in atomic_edges:

        if edge["orientation"] == "h":
            start = get_connector(
                edge["a"],
                edge["coord"]
            )
            end = get_connector(
                edge["b"],
                edge["coord"]
            )
        else:
            start = get_connector(
                edge["coord"],
                edge["a"]
            )
            end = get_connector(
                edge["coord"],
                edge["b"]
            )

        if start == end:
            continue

        pair_key = tuple(
            sorted([start, end])
        )

        if pair_key in segment_lookup:
            continue

        segment = copy.deepcopy(
            segment_template
        )

        segment["uuid"] = str(
            uuid.uuid4()
        )

        segment["start"] = start
        segment["end"] = end

        segment["thickness"] = (
            args.wall_thickness
        )

        segment["height"] = (
            args.height
        )

        segment["alignment"] = 0.5

        segment["left"] = connector_num(start)
        segment["right"] = connector_num(end)

        segment["half_width_left"] = (
            args.wall_thickness / 2.0
        )

        segment["half_width_right"] = (
            args.wall_thickness / 2.0
        )

        segment["cross_left"] = "sides"
        segment["cross_right"] = "sides"

        segment["align_left"] = "center"
        segment["align_right"] = "center"

        segment["userChangeBack"] = False
        segment["userChangeFront"] = False

        segments.append(segment)

        segment_lookup[pair_key] = (
            segment["uuid"]
        )

    # -------------------------------------------------
    # Adjacency
    # -------------------------------------------------

    conn_map = {
        c["uuid"]: c
        for c in connectors
    }

    for s in segments:
        a = s["start"]
        b = s["end"]

        if b not in conn_map[a]["adjacency"]:
            conn_map[a]["adjacency"].append(b)

        if a not in conn_map[b]["adjacency"]:
            conn_map[b]["adjacency"].append(a)

    # -------------------------------------------------
    # Area connector cycles
    #
    # Берём все connector'ы, лежащие на четырёх сторонах
    # прямоугольника комнаты.
    # -------------------------------------------------

    world_coords = {
        c["uuid"]: (
            float(c["x"]),
            float(c["y"])
        )
        for c in connectors
    }

    areas = []

    for room_index, r in enumerate(room_rects):

        x1w, y1w = to_world(
            r["x1"],
            r["y1"]
        )

        x2w, y2w = to_world(
            r["x2"],
            r["y2"]
        )

        tol = max(
            0.5,
            args.snap * args.scale
        )

        top = []
        right = []
        bottom = []
        left = []

        for cid, (x, y) in world_coords.items():

            if (
                abs(y - y1w) <= tol and
                min(x1w, x2w) - tol <= x <= max(x1w, x2w) + tol
            ):
                top.append(cid)

            if (
                abs(x - x2w) <= tol and
                min(y1w, y2w) - tol <= y <= max(y1w, y2w) + tol
            ):
                right.append(cid)

            if (
                abs(y - y2w) <= tol and
                min(x1w, x2w) - tol <= x <= max(x1w, x2w) + tol
            ):
                bottom.append(cid)

            if (
                abs(x - x1w) <= tol and
                min(y1w, y2w) - tol <= y <= max(y1w, y2w) + tol
            ):
                left.append(cid)

        top.sort(
            key=lambda cid: world_coords[cid][0]
        )

        right.sort(
            key=lambda cid: world_coords[cid][1]
        )

        bottom.sort(
            key=lambda cid: world_coords[cid][0],
            reverse=True
        )

        left.sort(
            key=lambda cid: world_coords[cid][1],
            reverse=True
        )

        cycle = []

        for side in (
            top,
            right,
            bottom,
            left
        ):
            for cid in side:
                if not cycle or cycle[-1] != cid:
                    cycle.append(cid)

        # Убираем повтор первого connector в конце
        if (
            len(cycle) > 1 and
            cycle[0] == cycle[-1]
        ):
            cycle.pop()

        # Убираем повторения вообще,
        # сохраняя порядок.
        seen = set()
        clean_cycle = []

        for cid in cycle:
            if cid not in seen:
                seen.add(cid)
                clean_cycle.append(cid)

        if len(clean_cycle) < 4:
            print(
                f"WARNING: Room {room_index + 1} "
                f"has only {len(clean_cycle)} connectors"
            )

        pts = [
            world_coords[cid]
            for cid in clean_cycle
        ]

        area = copy.deepcopy(
            area_template
        )

        area["uuid"] = str(
            uuid.uuid4()
        )

        area["name"] = (
            f"Room {room_index + 1}"
        )

        area["data"] = ""

        area["connectors"] = (
            clean_cycle
        )

        area["area_value"] = round(
            polygon_area(pts),
            4
        )

        areas.append(area)

    # -------------------------------------------------
    # Final project
    # -------------------------------------------------

    result = copy.deepcopy(
        template
    )

    level = result["levels"][0]

    level["height"] = args.height
    level["connectors"] = connectors
    level["segments"] = segments
    level["areas"] = areas

    # На первом тесте мебель/двери/окна не переносим.
    level["elements"] = []
    level["tapes"] = []

    result["editable"] = True

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with output_path.open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result,
            f,
            indent=2,
            ensure_ascii=False
        )

    if args.geometry_report or args.overlay:
        report = {"adapter": "bbox", "pixel_to_world": {
            "scale_x": args.scale, "scale_y": args.scale,
            "origin_x": origin_x - min_px_x * args.scale,
            "origin_y": origin_y - min_px_y * args.scale,
        }, "errors": geometry_errors, "warnings": [], "unsupported_segments": []}
        if args.geometry_report:
            report_path = Path(args.geometry_report)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        if args.overlay and args.source_image:
            from polygon_geometry import render_geometry
            render_geometry(result, report, args.source_image, args.overlay)

    print()
    print("Project JSON created")
    print(f"Output:     {output_path}")
    print()
    print(f"Rooms:      {len(areas)}")
    print(f"Connectors: {len(connectors)}")
    print(f"Segments:   {len(segments)}")
    print()
    print(f"Scale:      {args.scale} cm/px")
    print(f"Wall:       {args.wall_thickness} cm")
    print(f"Height:     {args.height} cm")
    print()


if __name__ == "__main__":
    main()
