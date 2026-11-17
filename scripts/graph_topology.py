"""Topology-only closures for physical wall graphs."""


def unresolved_gap_items(gaps, bridges):
    """Yield original indices and gaps whose endpoints still lack an opening bridge."""
    def endpoints(record):
        return tuple(sorted(tuple(point) for point in record['span']))

    # The carrier joins the graph endpoints. The clear span can be shorter by
    # both jamb widths. A partial overlap or a neighbouring jamb is ambiguous.
    occupied = {endpoints(bridge) for bridge in bridges}
    for index, gap in enumerate(gaps):
        if endpoints(gap) not in occupied:
            yield index, gap
