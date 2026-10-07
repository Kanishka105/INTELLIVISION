from pathlib import Path
import xml.etree.ElementTree as ET


# =========================================================
# PATHS
# =========================================================

CURRENT_FILE = Path(__file__).resolve()

PROJECT_DIR = CURRENT_FILE.parents[3]

PLATE_DATASET = (
    PROJECT_DIR
    / "dataset"
    / "plate"
)


# =========================================================
# CONVERT ONE XML
# =========================================================

def convert_xml(xml_path):

    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")

    if size is None:
        print(f"⚠️ No size information: {xml_path.name}")
        return False

    image_width = float(
        size.findtext("width")
    )

    image_height = float(
        size.findtext("height")
    )

    labels = []

    for obj in root.findall("object"):

        class_name = (
            obj.findtext("name", "")
            .strip()
            .lower()
        )

        if class_name != "number_plate":
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

        # Clamp
        xmin = max(
            0,
            min(xmin, image_width)
        )

        xmax = max(
            0,
            min(xmax, image_width)
        )

        ymin = max(
            0,
            min(ymin, image_height)
        )

        ymax = max(
            0,
            min(ymax, image_height)
        )

        if xmax <= xmin or ymax <= ymin:
            continue

        # VOC → YOLO
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

    if not labels:
        print(
            f"⚠️ No number_plate object: "
            f"{xml_path.name}"
        )
        return False

    txt_path = (
        xml_path.parent
        / f"{xml_path.stem}.txt"
    )

    txt_path.write_text(
        "\n".join(labels),
        encoding="utf-8"
    )

    return True


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 60)
    print("      XML → YOLO PLATE LABEL CONVERSION")
    print("=" * 60)

    if not PLATE_DATASET.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n{PLATE_DATASET}"
        )

    total_xml = 0
    converted = 0

    for split in [
        "train",
        "val",
        "test"
    ]:

        labels_dir = (
            PLATE_DATASET
            / split
            / "labels"
        )

        if not labels_dir.exists():

            print(
                f"\n⚠️ Missing: {labels_dir}"
            )

            continue

        xml_files = sorted(
            labels_dir.glob("*.xml")
        )

        print(
            f"\n{split.upper()}: "
            f"{len(xml_files)} XML files"
        )

        for xml_file in xml_files:

            total_xml += 1

            if convert_xml(xml_file):
                converted += 1

    print("\n" + "=" * 60)

    print(
        f"XML files found : {total_xml}"
    )

    print(
        f"Converted       : {converted}"
    )

    print("=" * 60)

    print(
        "\n✅ YOLO TXT labels created."
    )


if __name__ == "__main__":
    main()