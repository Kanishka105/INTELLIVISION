from pathlib import Path
import random
import shutil
import xml.etree.ElementTree as ET


# =========================================================
# PATHS
# =========================================================

CURRENT_FILE = Path(__file__).resolve()

PROJECT_DIR = CURRENT_FILE.parents[3]

SOURCE_DIR = Path(
    r"C:\Users\well\Downloads"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "dataset"
    / "plate"
)


# =========================================================
# SETTINGS
# =========================================================

random.seed(42)


# =========================================================
# XML -> YOLO
# =========================================================

def convert_xml_to_yolo(xml_path):

    root = ET.parse(
        xml_path
    ).getroot()

    size = root.find("size")

    if size is None:
        return []

    image_width = float(
        size.findtext("width")
    )

    image_height = float(
        size.findtext("height")
    )

    labels = []

    for obj in root.findall("object"):

        name = (
            obj.findtext("name", "")
            .strip()
            .lower()
        )

        if name != "number_plate":
            continue

        box = obj.find("bndbox")

        if box is None:
            continue

        xmin = float(
            box.findtext("xmin")
        )

        ymin = float(
            box.findtext("ymin")
        )

        xmax = float(
            box.findtext("xmax")
        )

        ymax = float(
            box.findtext("ymax")
        )

        x_center = (
            (xmin + xmax) / 2
        ) / image_width

        y_center = (
            (ymin + ymax) / 2
        ) / image_height

        width = (
            xmax - xmin
        ) / image_width

        height = (
            ymax - ymin
        ) / image_height

        labels.append(
            f"0 "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{width:.6f} "
            f"{height:.6f}"
        )

    return labels


# =========================================================
# CLEAN OLD DATASET
# =========================================================

def clean_dataset():

    if OUTPUT_DIR.exists():

        for split in [
            "train",
            "val",
            "test"
        ]:

            split_dir = (
                OUTPUT_DIR / split
            )

            if split_dir.exists():

                shutil.rmtree(
                    split_dir
                )


# =========================================================
# PREPARE DATASET
# =========================================================

def prepare_dataset():

    print("=" * 60)
    print("     CLEAN PLATE DATASET PREPARATION")
    print("=" * 60)

    clean_dataset()

    pairs = []

    for image_path in sorted(
        SOURCE_DIR.glob("image_*.jpg")
    ):

        xml_path = (
            SOURCE_DIR
            / f"{image_path.stem}.xml"
        )

        if xml_path.exists():

            pairs.append(
                (
                    image_path,
                    xml_path
                )
            )

    print(
        f"\nValid pairs found: {len(pairs)}"
    )

    if not pairs:

        raise RuntimeError(
            "No valid image/XML pairs found."
        )

    random.shuffle(
        pairs
    )

    total = len(pairs)

    train_count = int(
        total * 0.70
    )

    val_count = int(
        total * 0.15
    )

    train_pairs = pairs[
        :train_count
    ]

    val_pairs = pairs[
        train_count:
        train_count + val_count
    ]

    test_pairs = pairs[
        train_count + val_count:
    ]

    splits = {
        "train": train_pairs,
        "val": val_pairs,
        "test": test_pairs
    }

    total_written = 0

    for split, split_pairs in splits.items():

        images_dir = (
            OUTPUT_DIR
            / split
            / "images"
        )

        labels_dir = (
            OUTPUT_DIR
            / split
            / "labels"
        )

        images_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        labels_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        print(
            f"{split.upper()}: "
            f"{len(split_pairs)}"
        )

        for image_path, xml_path in split_pairs:

            labels = convert_xml_to_yolo(
                xml_path
            )

            if not labels:
                continue

            shutil.copy2(
                image_path,
                images_dir
                / image_path.name
            )

            label_path = (
                labels_dir
                / f"{image_path.stem}.txt"
            )

            label_path.write_text(
                "\n".join(labels),
                encoding="utf-8"
            )

            total_written += 1

    print("\n" + "=" * 60)

    print(
        f"Total valid pairs : {len(pairs)}"
    )

    print(
        f"Total written     : {total_written}"
    )

    print(
        "✅ CLEAN PLATE DATASET CREATED"
    )

    print("=" * 60)


if __name__ == "__main__":
    prepare_dataset()