import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "postprocessing" / "results"


def _paths(patient_id: str) -> dict[str, Path]:
    patient_root = PROJECT_ROOT / "data" / "brats2023" / patient_id
    return {
        "mri": patient_root / f"{patient_id}-t1n.nii.gz",
        "ground_truth": patient_root / f"{patient_id}-seg.nii.gz",
        "prediction": PROJECT_ROOT / "Attention_unet" / "results" / "predictions" / f"{patient_id}_prediction_original.nii.gz",
        "cleaned": PROJECT_ROOT / "postprocessing" / "postprocessed" / "cleaned_prediction_mask.nii.gz",
    }


def _load(path: Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing visualization input: {path}")
    image = nib.load(str(path))
    data = np.asarray(image.get_fdata(), dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D NIfTI volume: {path}")
    return data, image


def _normalise(image: np.ndarray) -> np.ndarray:
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros_like(image)
    low, high = np.percentile(finite, (1, 99))
    if high <= low:
        return np.zeros_like(image)
    return np.clip((image - low) / (high - low), 0.0, 1.0)


def _overlay(axis, image: np.ndarray, mask: np.ndarray, title: str, slice_index: int) -> None:
    axis.imshow(image[:, :, slice_index].T, cmap="gray", origin="lower")
    masked = np.ma.masked_where(mask[:, :, slice_index].T == 0, mask[:, :, slice_index].T)
    axis.imshow(masked, cmap="jet", alpha=0.48, origin="lower", vmin=1, vmax=3)
    axis.set_title(title)
    axis.axis("off")


def create_visualization(patient_id: str) -> list[Path]:
    paths = _paths(patient_id)
    mri, mri_image = _load(paths["mri"])
    prediction, prediction_image = _load(paths["prediction"])
    cleaned, cleaned_image = _load(paths["cleaned"])
    ground_truth = None
    if paths["ground_truth"].is_file():
        ground_truth, ground_truth_image = _load(paths["ground_truth"])
        if ground_truth.shape != mri.shape or not np.allclose(ground_truth_image.affine, mri_image.affine):
            raise ValueError("Ground-truth geometry does not match the MRI reference")

    for name, volume, image in (("prediction", prediction, prediction_image), ("cleaned", cleaned, cleaned_image)):
        if volume.shape != mri.shape or not np.allclose(image.affine, mri_image.affine):
            raise ValueError(f"{name} geometry does not match the MRI reference")

    tumor_mask = cleaned > 0
    slice_index = int(np.argmax(np.sum(tumor_mask, axis=(0, 1)))) if np.any(tumor_mask) else mri.shape[2] // 2
    image = _normalise(mri)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    overlay_path = OUTPUT_DIR / f"{patient_id}_segmentation_overlays.png"
    figure, axes = plt.subplots(1, 3 if ground_truth is not None else 2, figsize=(15, 5))
    axes = np.atleast_1d(axes)
    _overlay(axes[0], image, prediction.astype(np.uint8), "Original prediction", slice_index)
    _overlay(axes[1], image, cleaned.astype(np.uint8), "Cleaned prediction", slice_index)
    if ground_truth is not None:
        _overlay(axes[2], image, ground_truth.astype(np.uint8), "Ground truth", slice_index)
    figure.suptitle(f"AI/model segmentation overlays: {patient_id} | axial slice {slice_index}")
    figure.tight_layout()
    figure.savefig(overlay_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    regions_path = OUTPUT_DIR / f"{patient_id}_wt_tc_et.png"
    region_masks = [cleaned > 0, np.isin(cleaned, [1, 3]), cleaned == 3]
    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    for axis, region_mask, title in zip(axes, region_masks, ("WT", "TC", "ET")):
        _overlay(axis, image, region_mask.astype(np.uint8), title, slice_index)
    figure.suptitle(f"AI/model WT/TC/ET segmentation: {patient_id} | axial slice {slice_index}")
    figure.tight_layout()
    figure.savefig(regions_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return [overlay_path, regions_path]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create patient-specific MRI segmentation visualizations.")
    parser.add_argument("--patient-id", required=True)
    args = parser.parse_args()
    outputs = create_visualization(args.patient_id)
    for output in outputs:
        print(f"Saved visualization: {output.resolve()}")


if __name__ == "__main__":
    main()
