import sys
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def polygon_to_int(points):
    return np.array(
        [[int(round(x)), int(round(y))] for x, y in points],
        dtype=np.int32
    )


def fill_polygon(mask, polygon, value=255):
    outer = polygon.get("outer", [])

    if len(outer) < 3:
        return

    cv2.fillPoly(mask, [polygon_to_int(outer)], value)

    # Если внутри wall polygon есть holes —
    # возвращаем их в свободное пространство.
    for hole in polygon.get("holes", []):
        if len(hole) >= 3:
            cv2.fillPoly(mask, [polygon_to_int(hole)], 0)


def simplify_contour(contour):
    perimeter = cv2.arcLength(contour, True)

    # Умеренное упрощение контура комнаты
    epsilon = max(2.0, perimeter * 0.003)

    approx = cv2.approxPolyDP(
        contour,
        epsilon,
        True
    )

    return approx.reshape(-1, 2)


def main():
    if len(sys.argv) != 3:
        print(
            "Usage: python scripts/extract_rooms.py "
            "image.jpg classified.json"
        )
        sys.exit(1)

    image_path = Path(sys.argv[1])
    json_path = Path(sys.argv[2])

    if not image_path.exists():
        print("Image not found:", image_path)
        sys.exit(1)

    if not json_path.exists():
        print("JSON not found:", json_path)
        sys.exit(1)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta = data.get("meta", {})

    width = int(meta.get("width", 0))
    height = int(meta.get("height", 0))

    if width <= 0 or height <= 0:
        image = Image.open(image_path)
        width, height = image.size

    # --------------------------------------------------
    # 1. WALL MASK
    # --------------------------------------------------

    barrier = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    walls = data.get("walls", [])

    for wall in walls:
        fill_polygon(
            barrier,
            wall,
            255
        )

    # --------------------------------------------------
    # 2. CLOSE ALL OPENINGS
    # --------------------------------------------------
    #
    # Временно превращаем двери и окна обратно
    # в сплошную стену, чтобы каждая комната
    # стала замкнутой областью.
    # --------------------------------------------------

    openings = (
        data.get("doors", []) +
        data.get("windows", [])
    )

    opening_mask = np.zeros_like(barrier)

    for opening in openings:
        outer = opening.get("outer", [])

        if len(outer) >= 3:
            cv2.fillPoly(
                opening_mask,
                [polygon_to_int(outer)],
                255
            )

    # Немного расширяем проём,
    # чтобы гарантированно соединить его со стеной.
    opening_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (5, 5)
    )

    opening_mask = cv2.dilate(
        opening_mask,
        opening_kernel,
        iterations=1
    )

    barrier = cv2.bitwise_or(
        barrier,
        opening_mask
    )

    # --------------------------------------------------
    # 3. SEAL SMALL GAPS
    # --------------------------------------------------

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (5, 5)
    )

    barrier = cv2.morphologyEx(
        barrier,
        cv2.MORPH_CLOSE,
        close_kernel,
        iterations=1
    )

    # Совсем лёгкое расширение стены,
    # чтобы убрать 1–2 px разрывы.
    small_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, 3)
    )

    barrier = cv2.dilate(
        barrier,
        small_kernel,
        iterations=1
    )

    # --------------------------------------------------
    # 4. FREE SPACE
    # --------------------------------------------------

    free_space = cv2.bitwise_not(barrier)

    # --------------------------------------------------
    # 5. CONNECTED COMPONENTS
    # --------------------------------------------------

    count, labels, stats, centroids = \
        cv2.connectedComponentsWithStats(
            free_space,
            connectivity=8
        )

    rooms = []
    rejected = []

    image_area = width * height

    # Для нашего типа планов этого достаточно,
    # чтобы мелкий шум не становился комнатой.
    min_room_area = max(
        1500,
        int(image_area * 0.005)
    )

    room_id = 0

    for label_id in range(1, count):
        x, y, w, h, area = stats[label_id]

        # --------------------------------------------------
        # Компонент касается края изображения →
        # это наружный фон.
        # --------------------------------------------------

        touches_border = (
            x <= 1 or
            y <= 1 or
            x + w >= width - 1 or
            y + h >= height - 1
        )

        if touches_border:
            rejected.append({
                "label": int(label_id),
                "reason": "outside",
                "area_px": int(area)
            })
            continue

        if area < min_room_area:
            rejected.append({
                "label": int(label_id),
                "reason": "too_small",
                "area_px": int(area)
            })
            continue

        component = np.zeros(
            (height, width),
            dtype=np.uint8
        )

        component[labels == label_id] = 255

        contours, _ = cv2.findContours(
            component,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE
        )

        if not contours:
            continue

        contour = max(
            contours,
            key=cv2.contourArea
        )

        simplified = simplify_contour(
            contour
        )

        if len(simplified) < 3:
            continue

        cx, cy = centroids[label_id]

        polygon = [
            [
                int(point[0]),
                int(point[1])
            ]
            for point in simplified
        ]

        rooms.append({
            "id": room_id,
            "polygon": polygon,

            "center": [
                round(float(cx), 2),
                round(float(cy), 2)
            ],

            "bbox": [
                int(x),
                int(y),
                int(x + w),
                int(y + h)
            ],

            "area_px": int(area),

            "area_ratio": round(
                area / image_area,
                4
            )
        })

        room_id += 1

    # --------------------------------------------------
    # 6. RESULT
    # --------------------------------------------------

    result = {
        "meta": meta,

        "walls": walls,

        "doors": data.get(
            "doors",
            []
        ),

        "windows": data.get(
            "windows",
            []
        ),

        "rooms": rooms,

        "diagnostics": {
            "connected_components": count - 1,
            "rooms_found": len(rooms),
            "min_room_area": min_room_area,
            "rejected_components": rejected
        }
    }

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    output_json = (
        output_dir /
        f"{image_path.stem}_rooms.json"
    )

    with open(
        output_json,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
            ensure_ascii=False
        )

    # --------------------------------------------------
    # 7. DEBUG MASK
    # --------------------------------------------------

    cv2.imwrite(
        str(
            output_dir /
            f"{image_path.stem}_barrier.png"
        ),
        barrier
    )

    # --------------------------------------------------
    # 8. OVERLAY
    # --------------------------------------------------

    image = Image.open(
        image_path
    ).convert("RGB")

    draw = ImageDraw.Draw(image)

    # Walls = red
    for wall in walls:
        pts = [
            tuple(p)
            for p in wall.get("outer", [])
        ]

        if len(pts) >= 2:
            draw.line(
                pts + [pts[0]],
                fill="red",
                width=3
            )

    # Doors = green
    for opening in data.get("doors", []):
        pts = [
            tuple(p)
            for p in opening.get("outer", [])
        ]

        if len(pts) >= 2:
            draw.line(
                pts + [pts[0]],
                fill="green",
                width=5
            )

    # Windows = blue
    for opening in data.get("windows", []):
        pts = [
            tuple(p)
            for p in opening.get("outer", [])
        ]

        if len(pts) >= 2:
            draw.line(
                pts + [pts[0]],
                fill="blue",
                width=5
            )

    # Rooms
    for room in rooms:
        pts = [
            tuple(p)
            for p in room["polygon"]
        ]

        if len(pts) >= 2:
            draw.line(
                pts + [pts[0]],
                fill="magenta",
                width=4
            )

        cx, cy = room["center"]

        # Центр
        draw.ellipse(
            (
                cx - 5,
                cy - 5,
                cx + 5,
                cy + 5
            ),
            fill="magenta"
        )

        draw.text(
            (
                cx + 8,
                cy - 12
            ),
            f"Room {room['id'] + 1}",
            fill="magenta"
        )

    overlay_path = (
        output_dir /
        f"{image_path.stem}_rooms_overlay.png"
    )

    image.save(
        overlay_path
    )

    # --------------------------------------------------
    # 9. CONSOLE
    # --------------------------------------------------

    print()
    print("Room extraction done")
    print(f"JSON:     {output_json}")
    print(f"Overlay:  {overlay_path}")
    print()
    print(f"Walls:   {len(walls)}")
    print(f"Doors:   {len(data.get('doors', []))}")
    print(f"Windows: {len(data.get('windows', []))}")
    print(f"Rooms:   {len(rooms)}")
    print()

    for room in rooms:
        print(
            f"Room {room['id'] + 1}: "
            f"area={room['area_px']} px, "
            f"center={room['center']}"
        )


if __name__ == "__main__":
    main()
