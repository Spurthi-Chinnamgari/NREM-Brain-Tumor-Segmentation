"""Visualize a coordinate-reconstructed processed prediction."""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


CLASS_COLORS = ["black", "yellow", "purple", "red"]
CLASS_NAMES = ["Background", "NCR/NET", "Edema", "Enhancing Tumor"]


def load_volume(path: str) -> np.ndarray:
    volume = np.load(path)
    if volume.ndim == 4 and volume.shape[0] == 4:
        volume = volume[0]
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume or (4,X,Y,Z) MRI array, got {volume.shape}")
    return np.asarray(volume)


def select_slice(prediction: np.ndarray, requested: int | None) -> int:
    if requested is not None:
        if requested < 0 or requested >= prediction.shape[0]:
            raise ValueError(f"Slice {requested} is outside depth {prediction.shape[0]}")
        return requested
    tumor_counts = np.count_nonzero(prediction > 0, axis=(1, 2))
    return int(np.argmax(tumor_counts))


def save_visualization(
    prediction: np.ndarray,
    output_path: str,
    slice_index: int | None = None,
    image: np.ndarray | None = None,
) -> int:
    slice_index = select_slice(prediction, slice_index)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    figure, axis = plt.subplots(figsize=(8, 8))
    if image is not None:
        image_slice = image[slice_index]
        nonzero = image_slice[image_slice != 0]
        if nonzero.size:
            low, high = np.percentile(nonzero, [1, 99])
            axis.imshow(image_slice, cmap="gray", vmin=low, vmax=high)
    axis.imshow(
        prediction[slice_index],
        cmap=ListedColormap(CLASS_COLORS),
        vmin=0,
        vmax=3,
        alpha=0.65 if image is not None else 1.0,
        interpolation="nearest",
    )
    axis.set_title(f"Axial slice {slice_index}")
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return slice_index


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize a reconstructed processed BraTS prediction."
    )
    parser.add_argument("--prediction", required=True, help="Reconstructed prediction .npy file")
    parser.add_argument("--image", default=None, help="Optional matching 3D MRI .npy volume")
    parser.add_argument("--output", default="results/prediction.png")
    parser.add_argument("--slice", type=int, default=None)
    args = parser.parse_args()

    prediction = load_volume(args.prediction).astype(np.uint8)
    if not np.all(np.isin(prediction, [0, 1, 2, 3])):
        raise ValueError("Prediction contains labels outside 0..3")

    image = None
    if args.image:
        image = load_volume(args.image).astype(np.float32)
        if image.shape != prediction.shape:
            raise ValueError(f"MRI shape {image.shape} does not match prediction {prediction.shape}")

    slice_index = save_visualization(prediction, args.output, args.slice, image)
    print(f"Saved visualization: {args.output}")
    print(f"Slice: {slice_index}")
    print("Classes:", ", ".join(f"{index}={name}" for index, name in enumerate(CLASS_NAMES)))


if __name__ == "__main__":
    main()
