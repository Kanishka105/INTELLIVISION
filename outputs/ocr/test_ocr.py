from pathlib import Path
import sys


# =========================================================
# PATHS
# =========================================================

CURRENT_FILE = Path(__file__).resolve()

OCR_DIR = CURRENT_FILE.parent

if str(OCR_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(OCR_DIR)
    )


# =========================================================
# TEST OCR MODULES
# =========================================================

def test_imports():

    print("\n[1] Testing OCR modules...")

    from plate_crop import crop_plate
    from plate_text_cleaner import clean_plate_text

    assert callable(crop_plate)
    assert callable(clean_plate_text)

    print("✅ plate_crop imported")
    print("✅ plate_text_cleaner imported")


# =========================================================
# TEST PADDLEOCR
# =========================================================

def test_paddleocr():

    print("\n[2] Testing PaddleOCR...")

    from ocr_reader import OCRReader

    reader = OCRReader()

    assert reader is not None

    print("✅ PaddleOCR initialized")


# =========================================================
# TEST TEXT CLEANER
# =========================================================

def test_cleaner():

    print("\n[3] Testing text cleaner...")

    from plate_text_cleaner import clean_plate_text

    result = clean_plate_text(
        "DL 8C AF 1234"
    )

    print(
        "Input  : DL 8C AF 1234"
    )

    print(
        "Output :",
        result
    )

    assert result == "DL8CAF1234"

    print("✅ Text cleaner working")


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 50)
    print("       INTELLIVISION OCR TEST")
    print("=" * 50)

    test_imports()

    test_paddleocr()

    test_cleaner()

    print("\n" + "=" * 50)
    print("✅ OCR BASIC TEST PASSED")
    print("=" * 50)


if __name__ == "__main__":
    main()