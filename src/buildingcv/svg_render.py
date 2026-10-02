"""Render a CubiCasa SVG to an RGB image, dropping non-structural content.

The mask side of the dataset (svg_to_mask.py) only labels walls/doors/windows;
everything else collapses to `floor`. But cairosvg renders *all* SVG content
into the input image — furniture, fixtures, dimension lines, drains, text —
so the model is shown furniture and told it's floor. That input/label
mismatch is what's currently leaking onto wall predictions: thin black
furniture lines look like walls.

This module strips the non-structural subtrees *before* rendering, so the
input image carries only walls + doors + windows + open floor, matching the
mask's semantics.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

# Default-prefix the SVG namespace on serialization so the round-tripped
# document is plain `<svg xmlns="...">` rather than `<ns0:svg xmlns:ns0="...">`.
# cairosvg accepts both, but the default-namespace form is what the original
# files use and it keeps debug-saved SVGs human-readable.
ET.register_namespace("", "http://www.w3.org/2000/svg")

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


def render_input_png(svg_path: str | Path, out_size: tuple[int, int]) -> bytes:
    """Render the structural-only SVG to PNG bytes at (W, H).

    Output is pixel-aligned with `svg_to_mask.svg_to_mask` at the same size,
    so the (image, mask) pair stays consistent.
    """
    # Native Cairo is needed for SVG rendering, not for raster inference.
    import cairosvg

    W, H = out_size
    return cairosvg.svg2png(
        bytestring=filtered_svg_bytes(svg_path),
        output_width=W,
        output_height=H,
        background_color=_IMAGENET_MEAN_RGB,
    )
