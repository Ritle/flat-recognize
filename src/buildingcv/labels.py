"""Class taxonomy and the CubiCasa-token → class mapping.

Four classes. `floor` is the default fill (every pixel starts there).
Walls, doors, windows are painted on top in that order, so doors and windows
overwrite walls where they overlap (which matches the actual architecture —
a door is an opening *in* a wall).
"""

from collections import OrderedDict

CLASS_NAMES: tuple[str, ...] = ("floor", "wall", "door", "window")
CLASS_TO_ID: dict[str, int] = {n: i for i, n in enumerate(CLASS_NAMES)}
NUM_CLASSES: int = len(CLASS_NAMES)
FLOOR_ID: int = CLASS_TO_ID["floor"]
# Annotation-only sentinel; the model still predicts exactly four classes.
IGNORE_ID: int = 255

# Map: a single class token from a CubiCasa SVG → our class.
# Order = priority. When a polygon's effective class set (own + inherited)
# contains multiple keys, the first match wins. So a polygon nested as
# Wall > Door > Threshold gets classified as `door`, not `wall`.
TOKEN_TO_CLASS: "OrderedDict[str, str]" = OrderedDict(
    [
        ("Window", "window"),
        ("Door", "door"),
        ("Wall", "wall"),
    ]
)

# Order in which classes are painted onto the mask. Later overwrites earlier.
PAINT_ORDER: tuple[str, ...] = ("wall", "door", "window")

# RGB colors for visualization only — not used during training.
CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "floor": (240, 240, 235),
    "wall": (40, 40, 45),
    "door": (230, 120, 50),
    "window": (60, 150, 220),
}
