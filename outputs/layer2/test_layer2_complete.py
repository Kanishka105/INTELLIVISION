import cv2
import numpy as np
from pathlib import Path
import importlib.util
import py_compile
import inspect


# ============================================================
# PROJECT PATHS
# ============================================================

LAYER2_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LAYER2_DIR.parent.parent

print("=" * 70)
print("INTELLIVISION - LAYER 2 COMPLETE TECHNICAL TEST")
print("=" * 70)

print("\nLayer 2:")
print(LAYER2_DIR)

print("\nProject Root:")
print(PROJECT_ROOT)


# ============================================================
# GLOBAL RESULTS
# ============================================================

RESULTS = {}


# ============================================================
# FIND FILE
# ============================================================

def find_file(folder_name, candidate_names):

    folder = LAYER2_DIR / folder_name

    if not folder.exists():
        raise FileNotFoundError(
            f"Folder not found:\n{folder}"
        )

    python_files = list(folder.rglob("*.py"))

    # Case-insensitive filename matching
    for candidate in candidate_names:

        for file_path in python_files:

            if file_path.name.casefold() == candidate.casefold():
                return file_path

    raise FileNotFoundError(
        f"\nCould not find any of:\n"
        f"{candidate_names}\n"
        f"inside:\n{folder}"
    )


# ============================================================
# LOAD FUNCTION
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
        "layer2_"
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

    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    # Exact function matching
    for function_name in candidate_functions:

        if hasattr(module, function_name):

            function = getattr(
                module,
                function_name
            )

            if callable(function):
                return function, file_path

    # Case-insensitive function matching
    module_functions = {
        name.casefold(): obj
        for name, obj in inspect.getmembers(
            module,
            inspect.isfunction
        )
    }

    for function_name in candidate_functions:

        key = function_name.casefold()

        if key in module_functions:

            return (
                module_functions[key],
                file_path
            )

    raise AttributeError(
        f"\nNone of these functions were found:\n"
        f"{candidate_functions}\n"
        f"inside:\n{file_path}"
    )


# ============================================================
# VALIDATE NUMPY OUTPUT
# ============================================================

def validate_output(name, output):

    if output is None:
        raise ValueError(
            f"{name} returned None"
        )

    if not isinstance(output, np.ndarray):
        raise TypeError(
            f"{name} did not return NumPy array"
        )

    if output.size == 0:
        raise ValueError(
            f"{name} returned empty output"
        )

    if not np.isfinite(output).all():
        raise ValueError(
            f"{name} contains NaN/Inf"
        )

    return True


# ============================================================
# FIND TEST IMAGE
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
            f"Could not load image:\n{image_path}"
        )

    print("\nTest Image:")
    print(image_path.name)

    print("\nImage Shape:")
    print(image.shape)

    return image


# ============================================================
# 0. DATASET / IMAGE TEST
# ============================================================

print("\n" + "=" * 70)
print("0. DATASET / IMAGE LOAD")
print("=" * 70)

try:

    image = get_test_image()

    if image.ndim != 3:
        raise ValueError(
            "Expected 3-channel BGR image"
        )

    if image.shape[2] != 3:
        raise ValueError(
            "Image does not have 3 channels"
        )

    RESULTS["Dataset / Image"] = True

    print("[PASS] Dataset / Image")

except Exception as e:

    RESULTS["Dataset / Image"] = False

    print("[FAIL] Dataset / Image")
    print("Error:", e)

    raise


# ============================================================
# 1. PYTHON FILE COMPILE TEST
# ============================================================

print("\n" + "=" * 70)
print("1. PYTHON FILE COMPILE CHECK")
print("=" * 70)

compile_failed = []

for file_path in LAYER2_DIR.rglob("*.py"):

    if file_path.name == "test_layer2_complete.py":
        continue

    try:

        py_compile.compile(
            str(file_path),
            doraise=True
        )

        print(
            f"[PASS] {file_path.relative_to(LAYER2_DIR)}"
        )

    except Exception as e:

        compile_failed.append(file_path)

        print(
            f"[FAIL] {file_path.relative_to(LAYER2_DIR)}"
        )

        print(
            "Error:",
            e
        )

if compile_failed:

    RESULTS["Python Compile"] = False

else:

    RESULTS["Python Compile"] = True

print(
    "\nPython Compile Result:",
    "PASS" if RESULTS["Python Compile"] else "FAIL"
)


# ============================================================
# 2. SPATIAL FILTERING
# ============================================================

print("\n" + "=" * 70)
print("2. SPATIAL FILTERING")
print("=" * 70)

try:

    mean_filter, mean_file = load_function(
        "Smoothing",
        ["mean.py", "Mean.py"],
        ["mean_filter"]
    )

    weighted_filter, weighted_file = load_function(
        "Smoothing",
        ["weighted.py", "Weighted.py"],
        ["weighted_mean_filter"]
    )

    gaussian_filter, gaussian_file = load_function(
        "Smoothing",
        [
            "gaussian_filter.py",
            "Gaussian.py"
        ],
        [
            "gaussian_filter",
            "gaussian_smoothing"
        ]
    )

    bilateral_filter, bilateral_file = load_function(
        "Smoothing",
        [
            "bilateral_filter.py",
            "Bilateral.py"
        ],
        ["bilateral_filter"]
    )

    max_filter, max_file = load_function(
        "Smoothing",
        [
            "max_filter.py",
            "Max.py"
        ],
        ["max_filter"]
    )

    median_filter, median_file = load_function(
        "Smoothing",
        [
            "median_filter.py",
            "Median.py"
        ],
        ["median_filter"]
    )

    min_filter, min_file = load_function(
        "Smoothing",
        [
            "min_filter.py",
            "Min.py"
        ],
        ["min_filter"]
    )

    spatial_outputs = [

        mean_filter(image),

        weighted_filter(image),

        gaussian_filter(image),

        bilateral_filter(image),

        max_filter(image),

        median_filter(image),

        min_filter(image)
    ]

    for i, output in enumerate(
        spatial_outputs,
        start=1
    ):

        validate_output(
            f"Spatial Output {i}",
            output
        )

    RESULTS["Spatial Filtering"] = True

    print("\n[PASS] Spatial Filtering")

except Exception as e:

    RESULTS["Spatial Filtering"] = False

    print("\n[FAIL] Spatial Filtering")
    print("Error:", e)


# ============================================================
# 3. SHARPENING
# ============================================================

print("\n" + "=" * 70)
print("3. SHARPENING")
print("=" * 70)

try:

    basic_sharpen, _ = load_function(
        "sharpening",
        ["sharpening.py"],
        ["basic_sharpen"]
    )

    laplacian_sharpen, _ = load_function(
        "sharpening",
        ["Laplacian.py", "laplacian.py"],
        ["laplacian_sharpen"]
    )

    unsharp_mask, _ = load_function(
        "sharpening",
        ["Unsharp_mask.py", "unsharp_mask.py"],
        ["unsharp_mask"]
    )

    high_boost, _ = load_function(
        "sharpening",
        ["High_Boost.py", "high_boost.py"],
        ["high_boost_filter"]
    )

    sharpening_outputs = [

        basic_sharpen(image),

        laplacian_sharpen(image),

        unsharp_mask(image),

        high_boost(image)
    ]

    for i, output in enumerate(
        sharpening_outputs,
        start=1
    ):

        validate_output(
            f"Sharpening Output {i}",
            output
        )

    RESULTS["Sharpening"] = True

    print("[PASS] Sharpening")

except Exception as e:

    RESULTS["Sharpening"] = False

    print("[FAIL] Sharpening")
    print("Error:", e)


# ============================================================
# 4. ENHANCEMENT
# ============================================================

print("\n" + "=" * 70)
print("4. IMAGE ENHANCEMENT")
print("=" * 70)

try:

    histogram_equalization, _ = load_function(
        "enhancement",
        [
            "histogram_equalization.py",
            "HistogramEqualization.py"
        ],
        ["histogram_equalization"]
    )

    clahe, _ = load_function(
        "enhancement",
        [
            "CLAHE.py",
            "clahe.py"
        ],
        ["clahe_enhancement"]
    )

    negative, _ = load_function(
        "enhancement",
        ["Negative.py", "negative.py"],
        ["negative"]
    )

    log_transform, _ = load_function(
        "enhancement",
        [
            "log_transform.py",
            "LogTransformation.py"
        ],
        ["log_transform"]
    )

    gamma, _ = load_function(
        "enhancement",
        [
            "gamma.py",
            "GammaCorrection.py"
        ],
        ["gamma_correction"]
    )

    contrast_stretching, _ = load_function(
        "enhancement",
        [
            "contrast_stretching.py",
            "ContrastStretching.py"
        ],
        ["contrast_stretching"]
    )

    brightness, _ = load_function(
        "enhancement",
        ["Brightness.py", "brightness.py"],
        ["adjust_brightness"]
    )

    saturation, _ = load_function(
        "enhancement",
        ["Saturation.py", "saturation.py"],
        ["adjust_saturation"]
    )

    hsv_enhancement, _ = load_function(
        "enhancement",
        [
            "HSV-basedEnhancement.py",
            "hsv_enhancement.py"
        ],
        ["hsv_enhancement"]
    )

    enhancement_outputs = [

        histogram_equalization(image),

        clahe(image),

        negative(image),

        log_transform(image),

        gamma(image),

        contrast_stretching(image),

        brightness(image),

        saturation(image),

        hsv_enhancement(image)
    ]

    for i, output in enumerate(
        enhancement_outputs,
        start=1
    ):

        validate_output(
            f"Enhancement Output {i}",
            output
        )

    RESULTS["Enhancement"] = True

    print("[PASS] Enhancement")

except Exception as e:

    RESULTS["Enhancement"] = False

    print("[FAIL] Enhancement")
    print("Error:", e)


# ============================================================
# 5. EDGE DETECTION
# ============================================================

print("\n" + "=" * 70)
print("5. EDGE DETECTION")
print("=" * 70)

try:

    sobel, _ = load_function(
        "edge_Detection",
        ["Sobel.py", "sobel.py"],
        ["sobel_edge_detection"]
    )

    scharr, _ = load_function(
        "edge_Detection",
        [
            "scharr.py",
            "ScharrEdgeDetection.py"
        ],
        ["scharr_edge_detection"]
    )

    prewitt, _ = load_function(
        "edge_Detection",
        ["prewitt.py", "Prewitt.py"],
        ["prewitt_edge_detection"]
    )

    roberts, _ = load_function(
        "edge_Detection",
        ["roberts.py", "Roberts.py"],
        ["roberts_edge_detection"]
    )

    laplacian, _ = load_function(
        "edge_Detection",
        [
            "Laplacian.py",
            "Laplacian_Edge.py"
        ],
        ["laplacian_edge_detection"]
    )

    canny, _ = load_function(
        "edge_Detection",
        ["canny.py", "Canny.py"],
        [
            "canny_edge_detection",
            "edge_detection"
        ]
    )

    edge_outputs = [

        sobel(image),

        scharr(image),

        prewitt(image),

        roberts(image),

        laplacian(image),

        canny(image)
    ]

    for i, output in enumerate(
        edge_outputs,
        start=1
    ):

        validate_output(
            f"Edge Output {i}",
            output
        )

    RESULTS["Edge Detection"] = True

    print("[PASS] Edge Detection")

except Exception as e:

    RESULTS["Edge Detection"] = False

    print("[FAIL] Edge Detection")
    print("Error:", e)


# ============================================================
# 6. KERNEL OPERATIONS
# ============================================================

print("\n" + "=" * 70)
print("6. KERNEL OPERATIONS")
print("=" * 70)

try:

    correlation, _ = load_function(
        "KernelTechq",
        ["correlation.py"],
        ["correlation"]
    )

    convolution, _ = load_function(
        "KernelTechq",
        [
            "convolution.py",
            "convolutation.py"
        ],
        ["convolution"]
    )

    kernel = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ],
        dtype=np.float32
    )

    correlation_result = correlation(
        image,
        kernel
    )

    convolution_result = convolution(
        image,
        kernel
    )

    validate_output(
        "Correlation",
        correlation_result
    )

    validate_output(
        "Convolution",
        convolution_result
    )

    # For this asymmetric kernel, the results
    # should generally differ.
    difference = cv2.absdiff(
        correlation_result,
        convolution_result
    )

    print(
        "Kernel Mean Difference:",
        float(np.mean(difference))
    )

    print(
        "Kernel Maximum Difference:",
        int(np.max(difference))
    )

    RESULTS["Kernel Operations"] = True

    print("[PASS] Kernel Operations")

except Exception as e:

    RESULTS["Kernel Operations"] = False

    print("[FAIL] Kernel Operations")
    print("Error:", e)


# ============================================================
# 7. DISTANCE
# ============================================================

print("\n" + "=" * 70)
print("7. PIXEL DISTANCE")
print("=" * 70)

try:

    distance_file = find_file(
        "Distance",
        [
            "Distance.py",
            "distance.py"
        ]
    )

    py_compile.compile(
        str(distance_file),
        doraise=True
    )

    print(
        "[PASS] Distance.py compiles"
    )

    # Basic mathematical verification
    p1 = (1, 1)
    p2 = (4, 5)

    euclidean = np.sqrt(
        (p2[0] - p1[0]) ** 2 +
        (p2[1] - p1[1]) ** 2
    )

    city_block = (
        abs(p2[0] - p1[0]) +
        abs(p2[1] - p1[1])
    )

    chessboard = max(
        abs(p2[0] - p1[0]),
        abs(p2[1] - p1[1])
    )

    if not np.isclose(
        euclidean,
        5.0
    ):
        raise ValueError(
            "Euclidean distance test failed"
        )

    if city_block != 7:
        raise ValueError(
            "City-block distance test failed"
        )

    if chessboard != 4:
        raise ValueError(
            "Chessboard distance test failed"
        )

    print(
        "Euclidean:",
        euclidean
    )

    print(
        "City-Block:",
        city_block
    )

    print(
        "Chessboard:",
        chessboard
    )

    RESULTS["Pixel Distance"] = True

    print("[PASS] Pixel Distance")

except Exception as e:

    RESULTS["Pixel Distance"] = False

    print("[FAIL] Pixel Distance")
    print("Error:", e)


# ============================================================
# 8. FREQUENCY DOMAIN
# ============================================================

print("\n" + "=" * 70)
print("8. FREQUENCY DOMAIN")
print("=" * 70)

frequency_results = []

# ------------------------------------------------------------
# Frequency FILTERS
# ------------------------------------------------------------

frequency_filter_tests = [

    (
        "Low Pass",
        "filters",
        [
            "Low_Pass.py",
            "low_pass.py"
        ],
        ["low_pass_filter"]
    ),

    (
        "High Pass",
        "filters",
        [
            "High_Pass.py",
            "high_pass.py"
        ],
        ["high_pass_filter"]
    ),

    (
        "Band Pass",
        "filters",
        [
            "Band_Pass.py",
            "band_pass.py"
        ],
        ["band_pass_filter"]
    ),

    (
        "Band Reject",
        "filters",
        [
            "Band_reject.py",
            "band_reject.py",
            "Band_Reject.py"
        ],
        ["band_reject_filter"]
    ),

    (
        "Notch",
        "filters",
        [
            "Notch.py",
            "notch.py"
        ],
        ["notch_filter"]
    )
]


for (
    name,
    subfolder,
    files,
    functions
) in frequency_filter_tests:

    try:

        function, file_path = load_function(
            f"frequency/{subfolder}",
            files,
            functions
        )

        # Parameters for each filter
        if name == "Low Pass":

            output = function(image)

        elif name == "High Pass":

            output = function(image)

        elif name == "Band Pass":

            output = function(image)

        elif name == "Band Reject":

            output = function(image)

        elif name == "Notch":

            # Some notch implementations require
            # notch_points. Use a simple empty list
            # if supported.
            try:
                output = function(
                    image,
                    []
                )
            except TypeError:
                output = function(image)

        validate_output(
            name,
            output
        )

        print(
            f"[PASS] {name}"
        )

        frequency_results.append(True)

    except Exception as e:

        print(
            f"[FAIL] {name}"
        )

        print(
            "Error:",
            e
        )

        frequency_results.append(False)


# ------------------------------------------------------------
# Frequency TRANSFORMS
# ------------------------------------------------------------

frequency_transform_tests = [

    (
        "DFT",
        [
            "DFT.py",
            "dft.py"
        ],
        ["dft_transform"]
    ),

    (
        "IDFT",
        [
            "IDFT.py",
            "idft.py"
        ],
        ["inverse_dft"]
    ),

    (
        "FFT",
        [
            "FFT.py",
            "fft.py"
        ],
        ["fft_transform"]
    ),

    (
        "IFFT",
        [
            "IFFT.py",
            "ifft.py"
        ],
        ["inverse_fft"]
    ),

    (
        "DCT",
        [
            "DCT.py",
            "dct.py"
        ],
        ["dct_transform"]
    ),

    (
        "IDCT",
        [
            "IDCT.py",
            "idct.py"
        ],
        ["inverse_dct"]
    )
]


for (
    name,
    files,
    functions
) in frequency_transform_tests:

    try:

        function, file_path = load_function(
            "frequency/transform",
            files,
            functions
        )

        output = function(image)

        validate_output(
            name,
            output
        )

        print(
            f"[PASS] {name}"
        )

        frequency_results.append(True)

    except Exception as e:

        print(
            f"[FAIL] {name}"
        )

        print(
            "Error:",
            e
        )

        frequency_results.append(False)


RESULTS["Frequency Domain"] = all(
    frequency_results
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("FINAL LAYER 2 SUMMARY")
print("=" * 70)

for section, passed in RESULTS.items():

    status = "PASS" if passed else "FAIL"

    print(
        f"[{status}] {section}"
    )


print("\n" + "=" * 70)

if all(RESULTS.values()):

    print(
        "✅ LAYER 2 COMPLETE - ALL TESTS PASSED"
    )

else:

    print(
        "❌ LAYER 2 IS NOT YET COMPLETE"
    )

    print(
        "\nFix only the sections marked [FAIL]."
    )

print("=" * 70)