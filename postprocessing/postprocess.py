import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VALID_LABELS = {0, 1, 2, 3}
LABEL_PRIORITY = (3, 2, 1)


def _default_prediction_path(patient_id: str | None = None) -> Path:
    if patient_id is None:
        split_file = PROJECT_ROOT / "data" / "splits" / "test.txt"
        if not split_file.is_file():
            raise FileNotFoundError(f"Missing test split: {split_file}")
        with split_file.open("r", encoding="utf-8") as handle:
            patient_ids = [line.strip() for line in handle if line.strip()]
        if not patient_ids:
            raise ValueError(f"No patients found in test split: {split_file}")
        patient_id = patient_ids[0]
    return PROJECT_ROOT / "Attention_unet" / "results" / "predictions" / f"{patient_id}_prediction_original.nii.gz"


def _label_counts(mask: np.ndarray) -> dict[int, int]:
    counts = {}
    for label in sorted(VALID_LABELS):
        counts[label] = int(np.count_nonzero(mask == label))
    return counts


def load_prediction_mask(path: str | Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    mask_path = Path(path).expanduser().resolve()
    if not mask_path.is_file():
        raise FileNotFoundError(f"Prediction file does not exist: {mask_path}")
    if not ((mask_path.suffix == ".gz" and mask_path.name.endswith(".nii.gz")) or mask_path.suffix == ".nii"):
        raise ValueError(f"Prediction must be a NIfTI file: {mask_path}")

    image = nib.load(str(mask_path))
    data = np.asarray(image.get_fdata(dtype=np.float64), dtype=np.float64)

    if data.ndim != 3:
        raise ValueError(f"Prediction must be a 3D volume: {mask_path} (got shape {data.shape})")
    if not np.all(np.isfinite(data)):
        raise ValueError(f"Prediction contains NaN/Inf values: {mask_path}")

    rounded = np.rint(data)
    if not np.allclose(data, rounded, rtol=0.0, atol=1e-6):
        raise ValueError(
            "Prediction contains non-integer values; segmentation labels must be integer-like. "
            f"Path: {mask_path}"
        )

    integer_data = rounded.astype(np.int16)
    unexpected = sorted(set(np.unique(integer_data).tolist()) - VALID_LABELS)
    if unexpected:
        raise ValueError(
            f"Prediction contains unexpected labels {unexpected}; expected only {sorted(VALID_LABELS)}. "
            f"Path: {mask_path}"
        )

    counts = _label_counts(integer_data)
    print("Input prediction:", mask_path)
    print("Input shape:", integer_data.shape)
    print("Input labels:", sorted(set(np.unique(integer_data).tolist())))
    print("Input voxel counts:")
    for label in [0, 1, 2, 3]:
        print(f"  label {label}: {counts[label]}")

    return integer_data, image


def validate_cleaned_mask(original: np.ndarray, cleaned: np.ndarray) -> None:
    if cleaned.shape != original.shape:
        raise ValueError(
            "Cleaned prediction shape does not match original prediction shape: "
            f"cleaned={cleaned.shape}, original={original.shape}"
        )

    unexpected = sorted(set(np.unique(cleaned).tolist()) - VALID_LABELS)
    if unexpected:
        raise ValueError(f"Cleaned prediction contains unexpected labels {unexpected}; allowed labels are {sorted(VALID_LABELS)}")

    original_counts = _label_counts(original)
    cleaned_counts = _label_counts(cleaned)
    print("\nValidation before save:")
    print("Original voxel counts:")
    for label in [0, 1, 2, 3]:
        print(f"  label {label}: {original_counts[label]}")
    print("Cleaned voxel counts:")
    for label in [0, 1, 2, 3]:
        print(f"  label {label}: {cleaned_counts[label]}")

    if np.any(cleaned < 0) or np.any(cleaned > 3):
        raise ValueError("Cleaned prediction contains invalid labels outside the allowed BraTS range [0, 1, 2, 3].")

    if not np.all(np.isfinite(cleaned.astype(np.float64))):
        raise ValueError("Cleaned prediction contains NaN/Inf values.")


def remove_small_components(mask: np.ndarray, min_size: int = 100, apply_closing: bool = True) -> np.ndarray:
    """Remove tiny connected components while preserving the project's label semantics.

    The project consistently uses 0=background, 1=NCR/NET, 2=edema, 3=enhancing tumor.
    Each label is processed independently with connectivity=2, and then a deterministic
    priority rule resolves any overlap caused by morphological expansion.
    """
    if mask.ndim != 3:
        raise ValueError(f"Expected a 3D mask but received shape {mask.shape}")

    mask = np.asarray(mask, dtype=np.int16)
    min_size = max(1, int(min_size))
    structure = ndimage.generate_binary_structure(rank=3, connectivity=2)
    closing_structure = ndimage.generate_binary_structure(rank=3, connectivity=1)
    candidate_labels: dict[int, np.ndarray] = {}

    for label in (1, 2, 3):
        binary_mask = (mask == label)
        original_voxels = int(np.count_nonzero(binary_mask))
        print(f"\nLabel {label}: original voxels = {original_voxels}")

        if original_voxels == 0:
            candidate_labels[label] = np.zeros(mask.shape, dtype=bool)
            continue

        connected, n_components = ndimage.label(binary_mask, structure=structure)
        component_sizes = np.bincount(connected.ravel())
        valid_labels = np.where(component_sizes >= min_size)[0]
        valid_labels = valid_labels[valid_labels != 0]

        retained = np.isin(connected, valid_labels)
        if apply_closing and np.any(retained):
            closed = np.zeros_like(retained)
            for component_label in valid_labels:
                component_mask = connected == component_label
                closed |= ndimage.binary_closing(
                    component_mask,
                    structure=closing_structure,
                    iterations=1,
                )
            retained = closed

        retained_voxels = int(np.count_nonzero(retained))
        voxel_count_change = retained_voxels - original_voxels
        print(f"Label {label}: connected components = {n_components}")
        print(f"Label {label}: retained voxels = {retained_voxels}")
        print(f"Label {label}: voxel count change = {voxel_count_change:+d}")
        candidate_labels[label] = retained.astype(bool)

    resolved = np.zeros(mask.shape, dtype=np.uint8)
    occupied = np.zeros(mask.shape, dtype=bool)
    for label in LABEL_PRIORITY:
        active = candidate_labels.get(label, np.zeros(mask.shape, dtype=bool))
        new_voxels = active & ~occupied
        resolved[new_voxels] = label
        occupied |= active

    total_before = int(np.count_nonzero(mask > 0))
    total_after = int(np.count_nonzero(resolved > 0))
    print(f"\nTotal tumor voxels before = {total_before}")
    print(f"Total tumor voxels after = {total_after}")
    return resolved.astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove small 3D connected components from a full-volume BraTS prediction while preserving spatial metadata.")
    parser.add_argument("--prediction", type=str, default=None, help="Path to the reconstructed original-space prediction NIfTI.")
    parser.add_argument("--patient-id", type=str, default=None, help="Patient ID used if no prediction path is supplied.")
    parser.add_argument("--output", type=str, default=None, help="Output path for the cleaned NIfTI file.")
    parser.add_argument("--min-size", type=int, default=100, help="Minimum connected-component size to retain for each tumor label.")
    parser.add_argument("--closing", dest="apply_closing", action="store_true", default=False, help="Apply per-component binary closing.")
    parser.add_argument("--no-closing", dest="apply_closing", action="store_false", help="Disable the conservative binary closing step.")
    args = parser.parse_args()

    prediction_path = Path(args.prediction) if args.prediction else _default_prediction_path(args.patient_id)
    output_path = Path(args.output) if args.output else PROJECT_ROOT / "postprocessing" / "postprocessed" / "cleaned_prediction_mask.nii.gz"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    original, original_nii = load_prediction_mask(prediction_path)
    cleaned = remove_small_components(original, min_size=args.min_size, apply_closing=args.apply_closing)
    validate_cleaned_mask(original, cleaned)

    header = original_nii.header.copy()
    header.set_data_dtype(np.uint8)
    clean_nii = nib.Nifti1Image(cleaned.astype(np.uint8), original_nii.affine, header=header)
    nib.save(clean_nii, str(output_path))

    print("\nSaved cleaned prediction NIfTI to:")
    print(output_path)
    print("Affine and spatial metadata preserved from the original reconstructed volume.")
    print("Finished.")


if __name__ == "__main__":
    main()