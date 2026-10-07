from pathlib import Path


def file_size_kb(
    file_path
):
    """
    Return file size in KB.
    """

    size_bytes = Path(
        file_path
    ).stat().st_size

    return size_bytes / 1024


def compression_ratio(
    original_path,
    compressed_path
):
    """
    Original size / compressed size.
    """

    original_size = (
        file_size_kb(
            original_path
        )
    )

    compressed_size = (
        file_size_kb(
            compressed_path
        )
    )

    if compressed_size == 0:
        return 0.0

    return (
        original_size
        / compressed_size
    )


def size_reduction_percent(
    original_path,
    compressed_path
):
    """
    Percentage of file-size reduction.
    """

    original_size = (
        file_size_kb(
            original_path
        )
    )

    compressed_size = (
        file_size_kb(
            compressed_path
        )
    )

    if original_size == 0:
        return 0.0

    return (
        (
            original_size
            - compressed_size
        )
        / original_size
    ) * 100