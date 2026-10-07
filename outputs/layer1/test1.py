from pathlib import Path

from image_loader import load_image
from dataset_analysis import analyze_dataset
from image_analysis import (
    analyze_image,
    analyze_channels,
    analyze_color_spaces
)

from data_loader import create_dataloader


# =====================================================
# PROJECT PATH
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "train"
    / "images"
)


print("\n========================================")
print("        INTELLIVISION - LAYER 1")
print("========================================")

print("\nDataset path:")
print(IMAGE_DIR)

print("\nDataset exists:")
print(IMAGE_DIR.exists())


# =====================================================
# STEP 1 — FIND ONE IMAGE
# =====================================================

print("\n========================================")
print("STEP 1: SINGLE IMAGE ACQUISITION")
print("========================================")

image_files = [
    p for p in IMAGE_DIR.iterdir()
    if p.suffix.lower()
    in {".jpg", ".jpeg", ".png", ".bmp"}
]

if not image_files:

    raise ValueError(
        "No images found in dataset!"
    )


first_image = image_files[0]

print("\nSelected image:")
print(first_image)


# =====================================================
# STEP 2 — LOAD IMAGE
# =====================================================

image = load_image(
    first_image
)

print("\nImage loaded successfully!")

print(
    "Raw image shape:",
    image.shape
)


# =====================================================
# STEP 3 — IMAGE ANALYSIS
# =====================================================

print("\n========================================")
print("STEP 2: IMAGE STUDY")
print("========================================")

analyze_image(
    image
)


# =====================================================
# STEP 4 — CHANNEL ANALYSIS
# =====================================================

analyze_channels(
    image
)


# =====================================================
# STEP 5 — COLOR SPACE ANALYSIS
# =====================================================

rgb, gray, hsv = analyze_color_spaces(
    image
)


# =====================================================
# STEP 6 — BATCH DATA LOADING
# =====================================================

print("\n========================================")
print("STEP 3: BATCH DATA LOADING")
print("========================================")

loader = create_dataloader(
    image_dir=IMAGE_DIR,
    batch_size=8,
    shuffle=True,
    num_workers=0
)


# =====================================================
# STEP 7 — TEST FIRST BATCH
# =====================================================

for batch_number, (
    images,
    paths
) in enumerate(loader):

    print(
        f"\nBatch number : {batch_number + 1}"
    )

    print(
        f"Batch size   : {len(images)}"
    )

    print(
        f"First image shape : {images[0].shape}"
    )

    print("\nImages in batch:")

    for path in paths[:3]:

        print(
            "   ",
            path
        )

    # Only test first batch
    break


print("\n========================================")
print("       LAYER 1 TEST COMPLETED")
print("========================================")


# =====================================================
# STEP 8 — DATASET CHARACTERISTICS
# =====================================================

print("\n========================================")
print("STEP 4: DATASET CHARACTERISTICS")
print("========================================")

analyze_dataset(
    IMAGE_DIR
)