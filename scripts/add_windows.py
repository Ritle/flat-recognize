import argparse
import copy
import json
import uuid
from pathlib import Path


def find_window_template(data):
    for level in data.get("levels", []):
        for element in level.get("elements", []):
            if element.get("type") == "windowItem":
                return copy.deepcopy(element)

    return None


def scale_hole_shape(shape, sx, sy):
    result = []

    for point in shape:
        result.append({
            "x": round(float(point.get("x", 0)) * sx, 5),
            "y": round(float(point.get("y", 0)) * sy, 5),
        })

    return result


def scale_sill_shape(shape, sx):
    result = []

    for point in shape:
        result.append({
            "x": round(float(point.get("x", 0)) * sx, 5),
            "y": float(point.get("y", 0)),
        })

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "project_json",
        help="Project JSON already containing doors"
    )

    parser.add_argument(
        "window_bindings_json",
        help="window_bindings.json from add_openings.py"
    )

    parser.add_argument(
        "template_json",
        help="JSON containing a real windowItem"
    )

    parser.add_argument(
        "--output",
        default="output/floor2_project_final.json"
    )

    parser.add_argument(
        "--window-height",
        type=float,
        default=None,
        help="Override window height in cm"
    )

    parser.add_argument(
        "--window-z",
        type=float,
        default=None,
        help="Override template window position.z"
    )

    args = parser.parse_args()

    # --------------------------------------------------
    # Load files
    # --------------------------------------------------

    with open(
        args.project_json,
        "r",
        encoding="utf-8"
    ) as f:
        project = json.load(f)

    with open(
        args.window_bindings_json,
        "r",
        encoding="utf-8"
    ) as f:
        bindings_data = json.load(f)

    with open(
        args.template_json,
        "r",
        encoding="utf-8"
    ) as f:
        template_data = json.load(f)

    level = project["levels"][0]

    template = find_window_template(
        template_data
    )

    if template is None:
        raise RuntimeError(
            "No windowItem found in template JSON"
        )

    bindings = bindings_data.get(
        "windows",
        []
    )

    if not bindings:
        raise RuntimeError(
            "No windows found in window_bindings.json"
        )

    # --------------------------------------------------
    # Existing wall UUIDs
    # --------------------------------------------------

    valid_wall_uuids = {
        segment["uuid"]
        for segment in level.get("segments", [])
    }

    # --------------------------------------------------
    # Template dimensions
    # --------------------------------------------------

    template_width = float(
        template.get("size", {}).get("x", 80)
    )

    template_height = float(
        template.get("size", {}).get("y", 60)
    )

    window_height = (
        args.window_height
        if args.window_height is not None
        else template_height
    )

    template_z = float(
        template.get(
            "position",
            {}
        ).get("z", 170)
    )

    window_z = (
        args.window_z
        if args.window_z is not None
        else template_z
    )

    original_hole_shape = (
        template
        .get("wall", {})
        .get("holeShape", [])
    )

    original_sill_shape = (
        template
        .get("sill", {})
        .get("shape", [])
    )

    added = []

    # --------------------------------------------------
    # Generate windows
    # --------------------------------------------------

    for index, binding in enumerate(bindings, 1):

        wall_uuid = binding.get(
            "wall_uuid"
        )

        if wall_uuid not in valid_wall_uuids:
            print(
                f"WARNING: Window {index}: "
                f"wall {wall_uuid} not found"
            )
            continue

        width_cm = float(
            binding.get(
                "width_cm",
                template_width
            )
        )

        position_percent = float(
            binding.get(
                "position_percent",
                50
            )
        )

        # Не даём окну стать слишком маленьким
        # из-за шумов распознавания.
        width_cm = max(
            30.0,
            width_cm
        )

        window = copy.deepcopy(
            template
        )

        window_uuid = str(
            uuid.uuid4()
        )

        window["uuid"] = (
            window_uuid
        )

        window["type"] = (
            "windowItem"
        )

        window["name"] = (
            f"wall_item_window_{index}"
        )

        # ----------------------------------------------
        # Size
        # ----------------------------------------------

        window["size"] = {
            "x": round(width_cm, 3),
            "y": round(window_height, 3),
        }

        # ----------------------------------------------
        # Wall binding
        # ----------------------------------------------

        wall = window.setdefault(
            "wall",
            {}
        )

        wall["uuid"] = (
            wall_uuid
        )

        wall["position"] = round(
            position_percent,
            6
        )

        wall["alignment"] = 0.5
        wall["offset"] = 0
        wall["side"] = "left"

        # ----------------------------------------------
        # Scale template holeShape
        #
        # У исходного окна:
        # 80 cm -> ~ +/-0.4 m
        # 60 cm -> ~ 0.6 m
        #
        # Масштабируем именно существующий shape,
        # чтобы сохранить формат приложения.
        # ----------------------------------------------

        sx = (
            width_cm /
            template_width
        )

        sy = (
            window_height /
            template_height
        )

        if original_hole_shape:
            wall["holeShape"] = (
                scale_hole_shape(
                    original_hole_shape,
                    sx,
                    sy
                )
            )

        # ----------------------------------------------
        # Vertical position
        # ----------------------------------------------

        position = window.setdefault(
            "position",
            {}
        )

        position["z"] = round(
            window_z,
            3
        )

        # ----------------------------------------------
        # Sill
        # ----------------------------------------------

        if original_sill_shape:

            sill = window.setdefault(
                "sill",
                {}
            )

            sill["shape"] = (
                scale_sill_shape(
                    original_sill_shape,
                    sx
                )
            )

        # ----------------------------------------------
        # Scale actual window model
        # ----------------------------------------------

        model = window.get(
            "model",
            {}
        )

        model_size = model.get(
            "size",
            {}
        )

        model_width = float(
            model_size.get(
                "x",
                template_width
            )
            or template_width
        )

        model_height = float(
            model_size.get(
                "y",
                template_height
            )
            or template_height
        )

        window["scale"] = {
            "x": round(
                width_cm /
                model_width,
                8
            ),
            "y": round(
                window_height /
                model_height,
                8
            ),
            "z": 1,
        }

        # ----------------------------------------------
        # Defaults from template
        # ----------------------------------------------

        window.setdefault(
            "flip",
            {
                "x": False,
                "z": False
            }
        )

        window.setdefault(
            "recess",
            10
        )

        window.setdefault(
            "doorOpensOutwards",
            False
        )

        window.setdefault(
            "configuration",
            2
        )

        window["no_render"] = False

        # ----------------------------------------------
        # Important:
        # do not accidentally reuse door semantics
        # ----------------------------------------------

        window.pop(
            "door",
            None
        )

        level.setdefault(
            "elements",
            []
        ).append(
            window
        )

        added.append({
            "window": index,
            "uuid": window_uuid,
            "wall_uuid": wall_uuid,
            "position": round(
                position_percent,
                3
            ),
            "width_cm": round(
                width_cm,
                2
            ),
            "height_cm": round(
                window_height,
                2
            ),
            "z": round(
                window_z,
                2
            ),
            "distance": binding.get(
                "distance_to_wall_cm"
            )
        })

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

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
            project,
            f,
            indent=2,
            ensure_ascii=False
        )

    # --------------------------------------------------
    # Diagnostics
    # --------------------------------------------------

    doors_count = sum(
        e.get("type") == "doorItem"
        for e in level.get("elements", [])
    )

    windows_count = sum(
        e.get("type") == "windowItem"
        for e in level.get("elements", [])
    )

    print()
    print("DONE")
    print()
    print(f"Doors:   {doors_count}")
    print(f"Windows: {windows_count}")
    print()
    print(f"Output: {output_path}")
    print()

    for item in added:
        print(
            f"Window {item['window']}: "
            f"wall={item['wall_uuid']} "
            f"position={item['position']}% "
            f"size={item['width_cm']}x"
            f"{item['height_cm']} cm "
            f"z={item['z']} "
            f"distance={item['distance']}"
        )


if __name__ == "__main__":
    main()
