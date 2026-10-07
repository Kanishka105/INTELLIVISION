def component_centroid(
    centroids,
    label
):

    x, y = centroids[label]

    return (
        float(x),
        float(y)
    )