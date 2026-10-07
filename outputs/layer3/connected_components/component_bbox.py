def component_bbox(
    stats,
    label
):

    x = int(stats[label, 0])
    y = int(stats[label, 1])

    width = int(stats[label, 2])
    height = int(stats[label, 3])

    return {
        "x": x,
        "y": y,
        "width": width,
        "height": height
    }