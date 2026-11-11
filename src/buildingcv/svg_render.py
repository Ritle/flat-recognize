"""Render aligned CubiCasa inputs with an explicit choice of scene content.

The default structural view preserves historical training behavior. The opt-in
full view keeps furniture and annotations as potential negative examples while
the segmentation mask still labels only architectural structure.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

# Default-prefix the SVG namespace on serialization so the round-tripped
# document is plain `<svg xmlns="...">` rather than `<ns0:svg xmlns:ns0="...">`.
# cairosvg accepts both, but the default-namespace form is what the original
# files use and it keeps debug-saved SVGs human-readable.
ET.register_namespace("", "http://www.w3.org/2000/svg")

SVG_INPUT_MODES = ("structural", "full")

# Class tokens (space-separated values of the SVG `class` attribute) that
# identify subtrees the model should not see in the input image. A node is
# dropped if any of its own class tokens appears here — children are dropped
# transitively because we remove the whole subtree, not just the tagged node.
DROP_TOKENS: frozenset[str] = frozenset(
    {
        # Furniture and fixtures — the dominant source of wall-IoU damage.
        "FixedFurniture",
        "FixedFurnitureSet",
        # Plumbing/hardware bits that aren't structural:
        "Faucet",
        "Hanger",
        "Railing",
        "InnerDrain",
        "OuterDrain",
        "OuterCircle",
        # Drafting annotations: dimensions, labels, north-arrows:
        "Dimension",
        "DimensionMark",
        "Direction",
        "TextLabel",
        "Name",
        "SpaceDimensionsLabel",
        # Stray legend-style markers:
        "electricitySign",
    }
)


def _prune(elem: ET.Element) -> None:
    """Remove direct children whose `class` attr contains any DROP token, recursively."""
    for child in list(elem):
        tokens = (child.get("class") or "").split()
        if any(t in DROP_TOKENS for t in tokens):
            elem.remove(child)
        else:
            _prune(child)


def filtered_svg_bytes(svg_path: str | Path) -> bytes:
    """Parse `svg_path`, drop non-structural subtrees, return serialized bytes."""
    tree = ET.parse(svg_path)
    _prune(tree.getroot())
    return ET.tostring(tree.getroot())


# ImageNet mean expressed in 0–255 RGB. Used as the cairosvg background color
# so that pixels outside any drawn polygon match the letterbox padding fill —
# both "non-content" regions look identical to the model after normalization,
# and neither collides with the black used for walls.
_IMAGENET_MEAN_RGB = "rgb(124, 116, 104)"


def render_input_png(svg_path: str | Path, out_size: tuple[int, int], *,
                     input_mode: str = "structural") -> bytes:
    """Render a structural or full visible scene at (W, H).

    Output is pixel-aligned with `svg_to_mask.svg_to_mask` at the same size,
    so the (image, mask) pair stays consistent.
    """
    if input_mode not in SVG_INPUT_MODES:
        raise ValueError(f"Unknown SVG input mode: {input_mode}")
    # Native Cairo is needed for SVG rendering, not for raster inference.
    import cairosvg

    W, H = out_size
    return cairosvg.svg2png(
        bytestring=filtered_svg_bytes(svg_path) if input_mode == "structural" else Path(svg_path).read_bytes(),
        output_width=W,
        output_height=H,
        background_color=_IMAGENET_MEAN_RGB,
    )
