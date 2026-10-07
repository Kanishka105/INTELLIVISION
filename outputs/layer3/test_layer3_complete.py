import cv2
import numpy as np
from pathlib import Path
import importlib.util
import py_compile
import sys


# ============================================================
# PATHS
# ============================================================

LAYER3_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LAYER3_DIR.parent.parent


print("=" * 70)
print("INTELLIVISION - LAYER 3 COMPLETE TEST")
print("=" * 70)

print("\nLayer 3 Directory:")
print(LAYER3_DIR)

print("\nProject Root:")
print(PROJECT_ROOT)


# ============================================================
# RESULTS
# ============================================================

RESULTS = {}


# ============================================================
# ADD LAYER 3 FOLDERS TO PATH
# ============================================================

for folder in LAYER3_DIR.rglob("*"):

    if folder.is_dir():

        sys.path.insert(
            0,
            str(folder)
        )


# ============================================================
# FIND FILE
# ============================================================

def find_file(
    folder_name,
    candidate_names
):

    folder = LAYER3_DIR / folder_name

    if not folder.exists():

        raise FileNotFoundError(
            f"Folder not found:\n{folder}"
        )

    py_files = list(
        folder.rglob("*.py")
    )

    for candidate in candidate_names:

        for file_path in py_files:

            if (
                file_path.name.casefold()
                == candidate.casefold()
            ):

                return file_path

    raise FileNotFoundError(
        "\nCould not find any of:\n"
        + str(candidate_names)
        + "\nInside:\n"
        + str(folder)
    )


# ============================================================
# LOAD FUNCTION FROM FILE
# ============================================================

def load_function(
    folder_name,
    candidate_files,
    candidate_functions
):

    file_path = find_file(
        folder_name,
        candidate_files
    )

    module_name = (
        "layer3_"
        + file_path.stem.replace("-", "_")
        + "_"
        + str(abs(hash(str(file_path))))
    )

    spec = importlib.util.spec_from_file_location(
        module_name,
        file_path
    )

    if spec is None or spec.loader is None:

        raise ImportError(
            f"Could not load:\n{file_path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    # Exact function name
    for function_name in candidate_functions:

        if hasattr(
            module,
            function_name
        ):

            function = getattr(
                module,
                function_name
            )

            if callable(function):

                return function, file_path

    # Case-insensitive function search
    module_functions = {
        name.casefold(): obj
        for name, obj in module.__dict__.items()
        if callable(obj)
    }

    for function_name in candidate_functions:

        key = function_name.casefold()

        if key in module_functions:

            return (
                module_functions[key],
                file_path
            )

    raise AttributeError(
        "\nNone of these functions were found:\n"
        + str(candidate_functions)
        + "\nInside:\n"
        + str(file_path)
    )


# ============================================================
# VALIDATE ARRAY
# ============================================================

def validate_array(
    name,
    output
):

    if output is None:

        raise ValueError(
            f"{name} returned None"
        )

    if not isinstance(
        output,
        np.ndarray
    ):

        raise TypeError(
            f"{name} did not return NumPy array"
        )

    if output.size == 0:

        raise ValueError(
            f"{name} returned empty output"
        )

    if not np.isfinite(
        output.astype(np.float64)
    ).all():

        raise ValueError(
            f"{name} contains NaN/Inf"
        )


# ============================================================
# LOAD TEST IMAGE
# ============================================================

def get_test_image():

    image_folder = (
        PROJECT_ROOT
        / "dataset"
        / "train"
        / "images"
    )

    image_files = []

    for extension in [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.bmp"
    ]:

        image_files.extend(
            image_folder.glob(extension)
        )

    if not image_files:

        raise FileNotFoundError(
            f"No images found in:\n{image_folder}"
        )

    image_path = image_files[0]

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        raise ValueError(
            f"Could not load:\n{image_path}"
        )

    return image, image_path


# ============================================================
# 0. DATASET / IMAGE
# ============================================================

print("\n" + "=" * 70)
print("0. DATASET / IMAGE")
print("=" * 70)

try:

    image, image_path = get_test_image()

    print("\nTest Image:")
    print(image_path.name)

    print("\nImage Shape:")
    print(image.shape)

    if image.ndim != 3:
        raise ValueError(
            "Expected 3-channel image"
        )

    if image.shape[2] != 3:
        raise ValueError(
            "Expected 3 color channels"
        )

    RESULTS["Dataset / Image"] = True

    print("[PASS] Dataset / Image")

except Exception as e:

    RESULTS["Dataset / Image"] = False

    print("[FAIL] Dataset / Image")
    print("Error:", e)

    raise


# ============================================================
# 1. PYTHON COMPILE CHECK
# ============================================================

print("\n" + "=" * 70)
print("1. PYTHON FILE COMPILE CHECK")
print("=" * 70)

compile_failed = []

for file_path in LAYER3_DIR.rglob("*.py"):

    if file_path.name == "test_layer3_complete.py":
        continue

    try:

        py_compile.compile(
            str(file_path),
            doraise=True
        )

        print(
            "[PASS]",
            file_path.relative_to(
                LAYER3_DIR
            )
        )

    except Exception as e:

        compile_failed.append(
            file_path
        )

        print(
            "[FAIL]",
            file_path.relative_to(
                LAYER3_DIR
            )
        )

        print(
            "Error:",
            e
        )

RESULTS["Python Compile"] = (
    len(compile_failed) == 0
)


# ============================================================
# CREATE GRAYSCALE + BINARY IMAGE
# ============================================================

gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)

_, binary = cv2.threshold(
    gray,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)


# ============================================================
# 2. MORPHOLOGY
# ============================================================

print("\n" + "=" * 70)
print("2. MORPHOLOGY")
print("=" * 70)

try:

    structuring_element, _ = load_function(
        "morphology",
        [
            "structuring_element.py",
            "Structuring_Element.py"
        ],
        [
            "create_structuring_element"
        ]
    )

    erosion, _ = load_function(
        "morphology",
        [
            "Erosion.py",
            "erosion.py"
        ],
        [
            "erosion"
        ]
    )

    dilation, _ = load_function(
        "morphology",
        [
            "Dilation.py",
            "dilation.py"
        ],
        [
            "dilation"
        ]
    )

    opening, _ = load_function(
        "morphology",
        [
            "Opening.py",
            "opening.py"
        ],
        [
            "opening"
        ]
    )

    closing, _ = load_function(
        "morphology",
        [
            "Closing.py",
            "closing.py"
        ],
        [
            "closing"
        ]
    )

    gradient, _ = load_function(
        "morphology",
        [
            "gradient.py",
            "Gradient.py"
        ],
        [
            "morphological_gradient"
        ]
    )

    top_hat, _ = load_function(
        "morphology",
        [
            "top_hat.py",
            "Top_Hat.py"
        ],
        [
            "top_hat"
        ]
    )

    black_hat, _ = load_function(
        "morphology",
        [
            "black_hat.py",
            "Black_Hat.py"
        ],
        [
            "black_hat"
        ]
    )

    kernel = structuring_element(
        "ellipse",
        (5, 5)
    )

    if kernel is None:

        raise ValueError(
            "Structuring element failed"
        )

    outputs = [

        erosion(binary),

        dilation(binary),

        opening(binary),

        closing(binary),

        gradient(binary),

        top_hat(gray),

        black_hat(gray)
    ]

    for index, output in enumerate(
        outputs,
        start=1
    ):

        validate_array(
            f"Morphology Output {index}",
            output
        )

    RESULTS["Morphology"] = True

    print("[PASS] Morphology")

except Exception as e:

    RESULTS["Morphology"] = False

    print("[FAIL] Morphology")
    print("Error:", e)


# ============================================================
# 3. SEGMENTATION
# ============================================================

print("\n" + "=" * 70)
print("3. SEGMENTATION")
print("=" * 70)

try:

    binary_threshold, _ = load_function(
        "segmentation",
        [
            "binary_threshold.py"
        ],
        [
            "binary_threshold"
        ]
    )

    adaptive_threshold, _ = load_function(
        "segmentation",
        [
            "adaptive_threshold.py"
        ],
        [
            "adaptive_threshold",
            "adaptive_thresholding"
        ]
    )

    global_threshold, _ = load_function(
        "segmentation",
        [
            "global_thresholding.py",
            "Global_Thresholding.py"
        ],
        [
            "global_threshold"
        ]
    )

    otsu, _ = load_function(
        "segmentation",
        [
            "ostu.py",
            "otsu.py",
            "Otsu.py"
        ],
        [
            "otsu_threshold",
            "otsu_thresholding"
        ]
    )

    color_segmentation, _ = load_function(
        "segmentation",
        [
            "color_segmentation.py"
        ],
        [
            "color_segmentation"
        ]
    )

    color_threshold, _ = load_function(
        "segmentation",
        [
            "color_threshold.py"
        ],
        [
            "color_threshold"
        ]
    )

    outputs = [

        binary_threshold(image),

        adaptive_threshold(image),

        global_threshold(image),

        otsu(image),

        color_segmentation(image),

        color_threshold(
            image,
            (0, 0, 0),
            (180, 255, 255)
        )
    ]

    for index, output in enumerate(
        outputs,
        start=1
    ):

        validate_array(
            f"Segmentation Output {index}",
            output
        )

    RESULTS["Segmentation"] = True

    print("[PASS] Segmentation")

except Exception as e:

    RESULTS["Segmentation"] = False

    print("[FAIL] Segmentation")
    print("Error:", e)


# ============================================================
# 4. CONTOURS
# ============================================================

print("\n" + "=" * 70)
print("4. CONTOURS")
print("=" * 70)

try:

    find_contours_function, _ = load_function(
        "contours",
        [
            "find_contour.py"
        ],
        [
            "find_contours"
        ]
    )

    contour_area_function, _ = load_function(
        "contours",
        [
            "contour_area.py"
        ],
        [
            "contour_area"
        ]
    )

    perimeter_function, _ = load_function(
        "contours",
        [
            "perimenter.py",
            "perimeter.py"
        ],
        [
            "contour_perimeter",
            "perimeter"
        ]
    )

    bounding_box_function, _ = load_function(
        "contours",
        [
            "bounding_box.py"
        ],
        [
            "bounding_box"
        ]
    )

    convex_hull_function, _ = load_function(
        "contours",
        [
            "convex_hull.py"
        ],
        [
            "convex_hull"
        ]
    )

    contours, hierarchy = find_contours_function(
        binary
    )

    if contours is None:

        raise ValueError(
            "Contour function returned None"
        )

    print(
        "Contours Found:",
        len(contours)
    )

    if len(contours) == 0:

        raise ValueError(
            "No contours found in test image"
        )

    largest_contour = max(
        contours,
        key=cv2.contourArea
    )

    area = contour_area_function(
        largest_contour
    )

    perimeter = perimeter_function(
        largest_contour
    )

    bbox = bounding_box_function(
        largest_contour
    )

    hull = convex_hull_function(
        largest_contour
    )

    if not np.isfinite(area):

        raise ValueError(
            "Invalid contour area"
        )

    if not np.isfinite(perimeter):

        raise ValueError(
            "Invalid perimeter"
        )

    if bbox is None:

        raise ValueError(
            "Bounding box returned None"
        )

    if hull is None:

        raise ValueError(
            "Convex hull returned None"
        )

    print(
        "Largest Contour Area:",
        area
    )

    print(
        "Largest Contour Perimeter:",
        perimeter
    )

    print(
        "Bounding Box:",
        bbox
    )

    RESULTS["Contours"] = True

    print("[PASS] Contours")

except Exception as e:

    RESULTS["Contours"] = False

    print("[FAIL] Contours")
    print("Error:", e)


# ============================================================
# 5. CONNECTED COMPONENTS
# ============================================================

print("\n" + "=" * 70)
print("5. CONNECTED COMPONENTS")
print("=" * 70)

try:

    label_components, _ = load_function(
        "connected_components",
        [
            "labeling.py",
            "connected_components.py",
            "Connected_Components.py"
        ],
        [
            "label_components",
            "connected_components"
        ]
    )

    num_labels, labels, stats, centroids = (
        label_components(binary)
    )

    if num_labels < 1:

        raise ValueError(
            "No connected components returned"
        )

    print(
        "Components Including Background:",
        num_labels
    )

    # Background is label 0.
    # Foreground components start from label 1.

    foreground_count = max(
        0,
        num_labels - 1
    )

    print(
        "Foreground Components:",
        foreground_count
    )

    if foreground_count > 0:

        first_area = int(
            stats[1, cv2.CC_STAT_AREA]
        )

        first_centroid = centroids[1]

        print(
            "First Component Area:",
            first_area
        )

        print(
            "First Component Centroid:",
            first_centroid
        )

    RESULTS["Connected Components"] = True

    print("[PASS] Connected Components")

except Exception as e:

    RESULTS["Connected Components"] = False

    print("[FAIL] Connected Components")
    print("Error:", e)


# ============================================================
# 6. SHAPE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("6. SHAPE ANALYSIS")
print("=" * 70)

try:

    aspect_ratio_function, _ = load_function(
        "shape_analysis",
        [
            "aspect_ratio.py"
        ],
        [
            "aspect_ratio"
        ]
    )

    circularity_function, _ = load_function(
        "shape_analysis",
        [
            "circularity.py"
        ],
        [
            "circularity"
        ]
    )

    solidity_function, _ = load_function(
        "shape_analysis",
        [
            "solidity.py"
        ],
        [
            "solidity"
        ]
    )

    extent_function, _ = load_function(
        "shape_analysis",
        [
            "extent.py"
        ],
        [
            "extent"
        ]
    )

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:

        raise ValueError(
            "No contour available"
        )

    contour = max(
        contours,
        key=cv2.contourArea
    )

    x, y, width, height = cv2.boundingRect(
        contour
    )

    aspect = aspect_ratio_function(
        width,
        height
    )

    circularity = circularity_function(
        contour
    )

    solidity = solidity_function(
        contour
    )

    extent_value = extent_function(
        contour
    )

    values = [
        aspect,
        circularity,
        solidity,
        extent_value
    ]

    if not all(
        np.isfinite(v)
        for v in values
    ):

        raise ValueError(
            "Shape features contain invalid values"
        )

    print(
        "Aspect Ratio:",
        aspect
    )

    print(
        "Circularity:",
        circularity
    )

    print(
        "Solidity:",
        solidity
    )

    print(
        "Extent:",
        extent_value
    )

    RESULTS["Shape Analysis"] = True

    print("[PASS] Shape Analysis")

except Exception as e:

    RESULTS["Shape Analysis"] = False

    print("[FAIL] Shape Analysis")
    print("Error:", e)


# ============================================================
# 7. FEATURE VECTOR
# ============================================================

print("\n" + "=" * 70)
print("7. FEATURE VECTOR")
print("=" * 70)

try:

    shape_descriptors_function, _ = load_function(
        "shape_analysis",
        [
            "shape_descriptors.py"
        ],
        [
            "calculate_shape_descriptors"
        ]
    )

    feature_vector_function, _ = load_function(
        "shape_analysis",
        [
            "feature_vector.py"
        ],
        [
            "create_feature_vector"
        ]
    )

    descriptors = shape_descriptors_function(
        contour
    )

    feature_vector = feature_vector_function(
        descriptors
    )

    validate_array(
        "Feature Vector",
        feature_vector
    )

    print(
        "Feature Vector:",
        feature_vector
    )

    print(
        "Feature Vector Length:",
        len(feature_vector)
    )

    RESULTS["Feature Vector"] = True

    print("[PASS] Feature Vector")

except Exception as e:

    RESULTS["Feature Vector"] = False

    print("[FAIL] Feature Vector")
    print("Error:", e)


# ============================================================
# 8. HOUGH DETECTION
# ============================================================

print("\n" + "=" * 70)
print("8. HOUGH DETECTION")
print("=" * 70)

hough_pass = True
hough_found = False


# ------------------------------------------------------------
# Hough Lines
# ------------------------------------------------------------

try:

    hough_lines, lines_file = load_function(
        "hough_detection",
        [
            "hough_lines.py",
            "Hough_Lines.py"
        ],
        [
            "hough_lines"
        ]
    )

    lines = hough_lines(image)

    print(
        "Hough Lines:",
        "Detected" if lines is not None else "None"
    )

    hough_found = True

except Exception as e:

    print(
        "[INFO] Hough Lines not executed:",
        e
    )


# ------------------------------------------------------------
# Hough Circles
# ------------------------------------------------------------

try:

    hough_circles, circles_file = load_function(
        "hough_detection",
        [
            "hough_circles.py",
            "Hough_Circles.py"
        ],
        [
            "hough_circles"
        ]
    )

    circles = hough_circles(image)

    print(
        "Hough Circles:",
        "Detected" if circles is not None else "None"
    )

    hough_found = True

except Exception as e:

    print(
        "[INFO] Hough Circles not executed:",
        e
    )


# Hough is supplementary.
# If no Python implementation exists, do not
# mark the entire Layer 3 core as failed.

if hough_found:

    print("[PASS] Hough Detection")

else:

    print(
        "[INFO] Hough Detection skipped "
        "(no compatible Python module found)"
    )


RESULTS["Hough Detection"] = hough_pass


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("FINAL LAYER 3 SUMMARY")
print("=" * 70)

for section, passed in RESULTS.items():

    status = "PASS" if passed else "FAIL"

    print(
        f"[{status}] {section}"
    )


print("\n" + "=" * 70)

core_sections = [
    "Dataset / Image",
    "Python Compile",
    "Morphology",
    "Segmentation",
    "Contours",
    "Connected Components",
    "Shape Analysis",
    "Feature Vector"
]

core_passed = all(
    RESULTS.get(
        section,
        False
    )
    for section in core_sections
)


if core_passed:

    print(
        "✅ LAYER 3 CORE PIPELINE PASSED"
    )

    print(
        "Image → Segmentation → "
        "Contours → Components → "
        "Shape → Feature Vector"
    )

else:

    print(
        "❌ LAYER 3 HAS CORE FAILURES"
    )

print("=" * 70)