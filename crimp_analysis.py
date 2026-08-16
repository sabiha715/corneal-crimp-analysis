# ============================================================
# COLLAGEN FIBRE CRIMP ANALYSIS
#
# 1. Load Forward SHG image
# 2. Preprocess image
# 3. Select analysis patch
# 4. Estimate dominant fibre orientation
# 5. Semi-automatic single-fibre measurement for reference
# 6. Automatic whole-patch fibre tracing
# 7. Calculate fibre waviness / crimp
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

from aicspylibczi import CziFile

import matplotlib.pyplot as plt
import numpy as np

from skimage.exposure import equalize_adapthist
from skimage.feature import structure_tensor
from skimage.filters import sato
from skimage.graph import route_through_array

from scipy.ndimage import gaussian_filter, rotate
from scipy.signal import find_peaks, savgol_filter


# ============================================================
# ANALYSIS PARAMETERS
# ============================================================

# CZI image selection
SCENE = 0
TIME = 0
CHANNEL = 4         # Forward SHG
Z_SLICE = 5         # Choose image 

# Patch selection 
PATCH_SIZE = 256
CENTRE_Y = 500
CENTRE_X = 500

# Preprocessing
CLAHE_CLIP_LIMIT = 0.1
GAUSSIAN_SIGMA = 1
STRUCTURE_TENSOR_SIGMA = 3

# Ridge enhancement
SATO_SIGMAS = range(1, 4)

# Semi-automatic tracing
BRIGHTNESS_WEIGHT = 0.4
RIDGE_WEIGHT = 0.6
SMOOTHING_WINDOW = 11

# Automatic whole-patch tracing
PEAK_PERCENTILE = 75
MIN_PEAK_DISTANCE = 5
MAX_Y_JUMP = 2
MAX_GAP = 4

MIN_TRACK_LENGTH = 80
MIN_END_DISTANCE = 60
MAX_WAVINESS = 1.4


# ============================================================
# 1. LOAD CZI IMAGE
# ============================================================

czi = CziFile(
    "data/Rabbit_Central_cornea_low_pressure.czi"
)

image, _ = czi.read_image()

img = image[
    SCENE,
    TIME,
    CHANNEL,
    Z_SLICE,
    :,
    :
]

print("Image shape:", img.shape)
print("Data type:", img.dtype)


# ============================================================
# 2. IMAGE PREPROCESSING
# ============================================================

# Percentile normalisation reduces the effect of
# unusually bright or dark pixels.

p1 = np.percentile(
    img,
    1
)

p99 = np.percentile(
    img,
    99
)

img_norm = (
    img - p1
) / (
    p99 - p1
)

img_norm = np.clip(
    img_norm,
    0,
    1
)


# CLAHE improves local contrast.

img_clahe = equalize_adapthist(
    img_norm,
    clip_limit=CLAHE_CLIP_LIMIT
)


# Gaussian smoothing is used for robust
# orientation estimation.

img_smooth = gaussian_filter(
    img_clahe,
    sigma=GAUSSIAN_SIGMA
)


# ============================================================
# 3. STRUCTURE-TENSOR ORIENTATION
# ============================================================

Axx, Axy, Ayy = structure_tensor(
    img_smooth,
    sigma=STRUCTURE_TENSOR_SIGMA
)

orientation = 0.5 * np.arctan2(
    2 * Axy,
    Axx - Ayy
)
# ============================================================
# DISPLAY IMAGE PREPROCESSING
# ============================================================

orientation_deg = np.degrees(
    orientation
)

fig, axes = plt.subplots(
    1,
    4,
    figsize=(18, 5)
)

# Raw Forward SHG image
axes[0].imshow(
    img,
    cmap="gray"
)

axes[0].set_title(
    "Raw Forward SHG"
)


# CLAHE contrast enhancement
axes[1].imshow(
    img_clahe,
    cmap="gray"
)

axes[1].set_title(
    "CLAHE"
)


# Gaussian smoothing
axes[2].imshow(
    img_smooth,
    cmap="gray"
)

axes[2].set_title(
    "Gaussian Smoothed"
)


# Structure tensor orientation
orientation_plot = axes[3].imshow(
    orientation_deg,
    cmap="hsv",
    vmin=-90,
    vmax=90
)

axes[3].set_title(
    "Fibre Orientation"
)


# Remove axes from all images
for ax in axes:

    ax.axis(
        "off"
    )


plt.tight_layout()
plt.show()


# ============================================================
# 4. EXTRACT ANALYSIS PATCH
# ============================================================

half = PATCH_SIZE // 2


# Smoothed patch:
# used for orientation estimation and visualisation.

patch = img_smooth[
    CENTRE_Y-half:CENTRE_Y+half,
    CENTRE_X-half:CENTRE_X+half
]


orientation_patch = orientation[
    CENTRE_Y-half:CENTRE_Y+half,
    CENTRE_X-half:CENTRE_X+half
]


# ============================================================
# 5. MEAN PATCH ORIENTATION
# ============================================================

# Circular averaging is required because fibre orientation
# is axial: 0 degrees and 180 degrees represent the same
# direction.

mean_orientation = 0.5 * np.arctan2(
    np.mean(
        np.sin(
            2 * orientation_patch
        )
    ),
    np.mean(
        np.cos(
            2 * orientation_patch
        )
    )
)

mean_orientation_deg = np.degrees(
    mean_orientation
)

print(
    f"\nMean fibre orientation: "
    f"{mean_orientation_deg:.2f} degrees"
)


# ============================================================
# DISPLAY SELECTED PATCH AND ORIENTATION
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 6)
)

axes[0].imshow(
    patch,
    cmap="gray"
)

axes[0].set_title(
    "Selected Forward SHG Patch"
)

axes[0].axis(
    "off"
)


# Display local orientation vectors

step = 12

Y, X = np.mgrid[
    0:PATCH_SIZE:step,
    0:PATCH_SIZE:step
]

theta = orientation_patch[
    ::step,
    ::step
]

U = np.cos(
    theta
)

V = np.sin(
    theta
)

axes[1].imshow(
    patch,
    cmap="gray"
)

axes[1].quiver(
    X,
    Y,
    U,
    V,
    scale=20
)

axes[1].set_title(
    f"Structure-Tensor Orientation\n"
    f"Mean = {mean_orientation_deg:.2f} degrees"
)

axes[1].axis(
    "off"
)

plt.tight_layout()
plt.show()


# ============================================================
# 6. RIDGE ENHANCEMENT
# ============================================================

# Normalise tracing patch to 0-1.

patch_auto = patch.astype(
    float
)

patch_auto = (
    patch_auto
    - patch_auto.min()
) / (
    patch_auto.max()
    - patch_auto.min()
    + 1e-10
)


# Sato filtering enhances long bright ridge-like
# collagen structures.

ridge_response = sato(
    patch_auto,
    sigmas=SATO_SIGMAS,
    black_ridges=False
)

ridge_norm = (
    ridge_response
    - ridge_response.min()
) / (
    ridge_response.max()
    - ridge_response.min()
    + 1e-10
)


fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 6)
)

axes[0].imshow(
    patch_auto,
    cmap="gray"
)

axes[0].set_title(
    "Selected Patch"
)

axes[1].imshow(
    ridge_norm,
    cmap="gray"
)

axes[1].set_title(
    "Sato Ridge Enhancement"
)

for ax in axes:

    ax.axis(
        "off"
    )

plt.tight_layout()
plt.show()


# ============================================================
# 7. SEMI-AUTOMATIC SINGLE-FIBRE REFERENCE
# ============================================================

# The user selects only the start and end of one fibre.
# Minimum-cost path analysis then follows the most likely
# fibre route between those locations.

fig, ax = plt.subplots(
    figsize=(8, 8)
)

ax.imshow(
    patch_auto,
    cmap="gray"
)

ax.set_title(
    "Semi-Automatic Reference\n"
    "Click START and END of one collagen fibre"
)

ax.axis(
    "off"
)

endpoint_clicks = plt.ginput(
    n=2,
    timeout=0
)

plt.close(
    fig
)


if len(
    endpoint_clicks
) != 2:

    raise ValueError(
        "Exactly two fibre endpoints are required."
    )


start_x, start_y = endpoint_clicks[0]
end_x, end_y = endpoint_clicks[1]


start = (
    int(
        round(
            start_y
        )
    ),
    int(
        round(
            start_x
        )
    )
)

end = (
    int(
        round(
            end_y
        )
    ),
    int(
        round(
            end_x
        )
    )
)


# ------------------------------------------------------------
# Fibre-likelihood image
# ------------------------------------------------------------

fibre_score = (
    BRIGHTNESS_WEIGHT
    * patch_auto
    +
    RIDGE_WEIGHT
    * ridge_norm
)

fibre_score = np.clip(
    fibre_score,
    0,
    1
)


# High fibre score = low cost.

cost_image = 1 / (
    fibre_score
    + 0.05
)


# ------------------------------------------------------------
# Minimum-cost fibre path
# ------------------------------------------------------------

indices, total_cost = route_through_array(
    cost_image,
    start,
    end,
    fully_connected=True,
    geometric=True
)

automatic_path = np.array(
    indices
)

path_y = automatic_path[
    :,
    0
]

path_x = automatic_path[
    :,
    1
]


# ------------------------------------------------------------
# Smooth path
# ------------------------------------------------------------

window_length = SMOOTHING_WINDOW

if len(
    path_x
) <= window_length:

    window_length = (
        len(
            path_x
        )
        - 1
    )

if window_length % 2 == 0:

    window_length -= 1


if window_length >= 5:

    path_x_smooth = savgol_filter(
        path_x,
        window_length=window_length,
        polyorder=2
    )

    path_y_smooth = savgol_filter(
        path_y,
        window_length=window_length,
        polyorder=2
    )

else:

    path_x_smooth = path_x
    path_y_smooth = path_y


# ------------------------------------------------------------
# Calculate semi-automatic waviness
# ------------------------------------------------------------

dx = np.diff(
    path_x_smooth
)

dy = np.diff(
    path_y_smooth
)

semi_path_length = np.sum(
    np.sqrt(
        dx**2
        +
        dy**2
    )
)


semi_end_distance = np.sqrt(
    (
        path_x_smooth[-1]
        -
        path_x_smooth[0]
    )**2
    +
    (
        path_y_smooth[-1]
        -
        path_y_smooth[0]
    )**2
)


semi_waviness = (
    semi_path_length
    /
    semi_end_distance
)


print(
    "\n===================================="
)

print(
    "SEMI-AUTOMATIC REFERENCE"
)

print(
    "===================================="
)

print(
    f"Fibre path length: "
    f"{semi_path_length:.2f} pixels"
)

print(
    f"End-to-end distance: "
    f"{semi_end_distance:.2f} pixels"
)

print(
    f"Waviness ratio: "
    f"{semi_waviness:.4f}"
)

print(
    "===================================="
)


# ------------------------------------------------------------
# Display semi-automatic result
# ------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(8, 8)
)

ax.imshow(
    patch_auto,
    cmap="gray"
)

ax.plot(
    path_x_smooth,
    path_y_smooth,
    linewidth=2,
    label="Automatic fibre path"
)

ax.plot(
    [
        path_x_smooth[0],
        path_x_smooth[-1]
    ],
    [
        path_y_smooth[0],
        path_y_smooth[-1]
    ],
    "--",
    linewidth=2,
    label="End-to-end distance"
)

ax.set_title(
    f"Semi-Automatic Fibre Measurement\n"
    f"Waviness = {semi_waviness:.4f}"
)

ax.legend()

ax.axis(
    "off"
)

plt.tight_layout()
plt.show()


# ============================================================
# 8. AUTOMATIC WHOLE-PATCH ANALYSIS
# ============================================================

# Rotate the patch so that the dominant fibres run
# approximately horizontally.

rotation_angle = (
    -mean_orientation_deg
)

patch_rotated = rotate(
    patch_auto,
    rotation_angle,
    reshape=True,
    order=1,
    mode="constant",
    cval=0
)

ridge_rotated = rotate(
    ridge_norm,
    rotation_angle,
    reshape=True,
    order=1,
    mode="constant",
    cval=0
)


print(
    f"\nPatch rotation: "
    f"{rotation_angle:.2f} degrees"
)


fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 6)
)

axes[0].imshow(
    patch_rotated,
    cmap="gray"
)

axes[0].set_title(
    "Rotated Patch"
)

axes[1].imshow(
    ridge_rotated,
    cmap="gray"
)

axes[1].set_title(
    "Rotated Ridge Image"
)

for ax in axes:

    ax.axis(
        "off"
    )

plt.tight_layout()
plt.show()


# ============================================================
# 9. AUTOMATIC RIDGE PEAK DETECTION
# ============================================================

valid_ridge_pixels = ridge_rotated[
    ridge_rotated > 0
]

peak_threshold = np.percentile(
    valid_ridge_pixels,
    PEAK_PERCENTILE
)


height, width = ridge_rotated.shape

column_peaks = []


for x in range(
    width
):

    column = ridge_rotated[
        :,
        x
    ]

    peaks, _ = find_peaks(
        column,
        height=peak_threshold,
        distance=MIN_PEAK_DISTANCE
    )

    column_peaks.append(
        peaks
    )


# ============================================================
# 10. LINK RIDGE PEAKS BETWEEN COLUMNS
# ============================================================

active_tracks = []
finished_tracks = []


for x in range(
    width
):

    peaks = column_peaks[
        x
    ]

    matched_peaks = set()


    # Try to extend existing tracks

    for track in active_tracks:

        last_x, last_y = (
            track[
                "points"
            ][-1]
        )


        candidate_indices = []
        candidate_distances = []


        for peak_index, peak_y in enumerate(
            peaks
        ):

            if peak_index in matched_peaks:

                continue


            distance = abs(
                peak_y
                -
                last_y
            )


            if distance <= MAX_Y_JUMP:

                candidate_indices.append(
                    peak_index
                )

                candidate_distances.append(
                    distance
                )


        if len(
            candidate_indices
        ) > 0:

            best_index = np.argmin(
                candidate_distances
            )

            peak_index = candidate_indices[
                best_index
            ]

            peak_y = peaks[
                peak_index
            ]


            track[
                "points"
            ].append(
                (
                    x,
                    peak_y
                )
            )

            track[
                "gap"
            ] = 0

            matched_peaks.add(
                peak_index
            )


        else:

            track[
                "gap"
            ] += 1


    # Finish tracks that have disappeared

    remaining_tracks = []


    for track in active_tracks:

        if track[
            "gap"
        ] > MAX_GAP:

            finished_tracks.append(
                track
            )

        else:

            remaining_tracks.append(
                track
            )


    active_tracks = remaining_tracks


    # Start new tracks using unused peaks

    for peak_index, peak_y in enumerate(
        peaks
    ):

        if peak_index in matched_peaks:

            continue


        active_tracks.append(
            {
                "points": [
                    (
                        x,
                        peak_y
                    )
                ],

                "gap": 0
            }
        )


finished_tracks.extend(
    active_tracks
)


print(
    f"\nRaw candidate tracks: "
    f"{len(finished_tracks)}"
)


# ============================================================
# 11. FILTER, SMOOTH AND MEASURE AUTOMATIC TRACKS
# ============================================================

valid_tracks = []
results = []


for track in finished_tracks:

    points = np.array(
        track[
            "points"
        ],
        dtype=float
    )


    # Reject short tracks

    if len(
        points
    ) < MIN_TRACK_LENGTH:

        continue


    x = points[
        :,
        0
    ]

    y = points[
        :,
        1
    ]


    # --------------------------------------------------------
    # Smooth fibre path
    # --------------------------------------------------------

    window_length = SMOOTHING_WINDOW

    if len(
        y
    ) <= window_length:

        window_length = (
            len(
                y
            )
            - 1
        )

    if window_length % 2 == 0:

        window_length -= 1


    if window_length >= 5:

        y_smooth = savgol_filter(
            y,
            window_length=window_length,
            polyorder=2
        )

    else:

        y_smooth = y


    # --------------------------------------------------------
    # Fibre path length
    # --------------------------------------------------------

    dx = np.diff(
        x
    )

    dy = np.diff(
        y_smooth
    )


    path_length = np.sum(
        np.sqrt(
            dx**2
            +
            dy**2
        )
    )


    # --------------------------------------------------------
    # End-to-end distance
    # --------------------------------------------------------

    end_distance = np.sqrt(
        (
            x[-1]
            -
            x[0]
        )**2
        +
        (
            y_smooth[-1]
            -
            y_smooth[0]
        )**2
    )


    if end_distance < MIN_END_DISTANCE:

        continue


    # --------------------------------------------------------
    # Waviness ratio
    # --------------------------------------------------------

    waviness = (
        path_length
        /
        end_distance
    )


    if not (
        1
        <= waviness
        <= MAX_WAVINESS
    ):

        continue


    valid_tracks.append(
        np.column_stack(
            (
                x,
                y_smooth
            )
        )
    )


    results.append(
        {
            "length": path_length,
            "end_to_end": end_distance,
            "waviness": waviness
        }
    )


# ============================================================
# 12. AUTOMATIC RESULTS
# ============================================================

print(
    f"Accepted fibres: "
    f"{len(results)}"
)


if len(
    results
) > 0:

    waviness_values = np.array(
        [
            result[
                "waviness"
            ]
            for result
            in results
        ]
    )


    print(
        "\n===================================="
    )

    print(
        "AUTOMATIC WHOLE-PATCH CRIMP"
    )

    print(
        "===================================="
    )


    for number, result in enumerate(
        results,
        start=1
    ):

        print(
            f"Fibre {number}: "
            f"Length = "
            f"{result['length']:.2f} px | "
            f"End-to-end = "
            f"{result['end_to_end']:.2f} px | "
            f"Waviness = "
            f"{result['waviness']:.4f}"
        )


    print(
        "\n------------------------------------"
    )

    print(
        f"Fibres measured: "
        f"{len(waviness_values)}"
    )

    print(
        f"Mean waviness: "
        f"{np.mean(waviness_values):.4f}"
    )

    print(
        f"Median waviness: "
        f"{np.median(waviness_values):.4f}"
    )

    print(
        f"Standard deviation: "
        f"{np.std(waviness_values):.4f}"
    )

    print(
        f"Minimum waviness: "
        f"{np.min(waviness_values):.4f}"
    )

    print(
        f"Maximum waviness: "
        f"{np.max(waviness_values):.4f}"
    )

    print(
        "===================================="
    )


# ============================================================
# 13. DISPLAY AUTOMATIC FIBRE TRACKS
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 10)
)

ax.imshow(
    patch_rotated,
    cmap="gray"
)


for number, track in enumerate(
    valid_tracks,
    start=1
):

    x = track[
        :,
        0
    ]

    y = track[
        :,
        1
    ]


    ax.plot(
        x,
        y,
        linewidth=2
    )


    midpoint = (
        len(
            track
        )
        // 2
    )


    ax.text(
        x[
            midpoint
        ],
        y[
            midpoint
        ],
        str(
            number
        ),
        fontsize=8
    )


ax.set_title(
    f"Automatic Fibre Tracks - Rotated Patch\n"
    f"N = {len(valid_tracks)}"
)

ax.axis(
    "off"
)

plt.tight_layout()
plt.show()


