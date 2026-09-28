import argparse
from pathlib import Path

import numpy as np
import nibabel as nib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VALID_LABELS = {0, 1, 2, 3}


def default_prediction_path(patient_id: str | None = None) -> Path:
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


def default_ground_truth_path(patient_id: str | None = None) -> Path:
    if patient_id is None:
        split_file = PROJECT_ROOT / "data" / "splits" / "test.txt"
        if not split_file.is_file():
            raise FileNotFoundError(f"Missing test split: {split_file}")
        with split_file.open("r", encoding="utf-8") as handle:
            patient_ids = [line.strip() for line in handle if line.strip()]
        if not patient_ids:
            raise ValueError(f"No patients found in test split: {split_file}")
        patient_id = patient_ids[0]
    return PROJECT_ROOT / "data" / "brats2023" / patient_id / f"{patient_id}-seg.nii.gz"


def load_mask(path: str | Path, name: str) -> tuple[np.ndarray, nib.Nifti1Image]:
    mask_path = Path(path).expanduser().resolve()
    if not mask_path.is_file():
        raise FileNotFoundError(f"{name} mask does not exist: {mask_path}")

    suffixes = mask_path.suffixes
    if not ((len(suffixes) >= 2 and suffixes[-2] == ".nii" and suffixes[-1] == ".gz") or mask_path.suffix == ".nii"):
        raise ValueError(f"{name} mask must be a NIfTI file: {mask_path}")

    image = nib.load(str(mask_path))
    data = np.asarray(image.get_fdata(dtype=np.float64), dtype=np.float64)

    if data.ndim < 3:
        raise ValueError(f"{name} mask is not volumetric: {mask_path}")
    if not np.all(np.isfinite(data)):
        raise ValueError(f"{name} mask contains NaN or Inf values: {mask_path}")

    rounded = np.rint(data)
    if not np.allclose(data, rounded, rtol=0.0, atol=1e-6):
        raise ValueError(
            f"{name} mask contains non-integer labels; segmentation values must be integer-like. "
            f"Path: {mask_path}"
        )

    integer_data = rounded.astype(np.int16)
    labels = set(np.unique(integer_data).tolist())
    unexpected = sorted(labels - VALID_LABELS)
    if unexpected:
        raise ValueError(
            f"{name} mask contains unexpected labels {unexpected}; expected only {sorted(VALID_LABELS)}. "
            f"Path: {mask_path}"
        )

    print(f"PASS: {name} loaded: {mask_path}")
    print(f"PASS: {name} shape = {integer_data.shape}")
    print(f"PASS: {name} labels = {sorted(labels)}")
    return integer_data, image


def validate_prediction_ground_truth(
    prediction_path: str | Path | None = None,
    ground_truth_path: str | Path | None = None,
    patient_id: str | None = None,
) -> tuple[np.ndarray, np.ndarray, nib.Nifti1Image, nib.Nifti1Image]:
    if prediction_path is None:
        prediction_path = default_prediction_path(patient_id)
    if ground_truth_path is None:
        ground_truth_path = default_ground_truth_path(patient_id)

    prediction, prediction_nii = load_mask(prediction_path, "Prediction")
    ground_truth, ground_truth_nii = load_mask(ground_truth_path, "Ground truth")

    if prediction.shape != ground_truth.shape:
        raise ValueError(
            "ERROR: Prediction and ground truth are not spatially compatible; shapes differ: "
            f"prediction={prediction.shape}, ground_truth={ground_truth.shape}. "
            f"Prediction={prediction_path}, Ground truth={ground_truth_path}"
        )

    prediction_spacing = tuple(float(v) for v in prediction_nii.header.get_zooms()[:3])
    ground_truth_spacing = tuple(float(v) for v in ground_truth_nii.header.get_zooms()[:3])
    if not np.allclose(prediction_spacing, ground_truth_spacing, rtol=1e-5, atol=1e-6):
        raise ValueError(
            "ERROR: Prediction and ground truth voxel spacing are incompatible: "
            f"prediction={prediction_spacing}, ground_truth={ground_truth_spacing}. "
            f"Prediction={prediction_path}, Ground truth={ground_truth_path}"
        )

    if not np.allclose(prediction_nii.affine, ground_truth_nii.affine, rtol=1e-5, atol=1e-6):
        raise ValueError(
            "ERROR: Prediction and ground truth affines are incompatible: "
            f"prediction_affine={prediction_nii.affine}, ground_truth_affine={ground_truth_nii.affine}. "
            f"Prediction={prediction_path}, Ground truth={ground_truth_path}"
        )

    print("PASS: Prediction and ground truth have identical 3D shape.")
    print(f"PASS: Prediction spacing = {prediction_spacing}")
    print(f"PASS: Ground-truth spacing = {ground_truth_spacing}")
    print("PASS: Prediction and ground truth affine matrices are compatible.")
    return prediction, ground_truth, prediction_nii, ground_truth_nii


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the original-space prediction and ground-truth NIfTI masks before metrics or postprocessing.")
    parser.add_argument("--patient-id", type=str, default=None, help="Patient ID used to resolve default prediction and ground-truth outputs.")
    parser.add_argument("--prediction", type=str, default=None, help="Path to the reconstructed original-space prediction NIfTI.")
    parser.add_argument("--ground-truth", type=str, default=None, help="Path to the ground-truth NIfTI.")
    args = parser.parse_args()

    try:
        validate_prediction_ground_truth(args.prediction, args.ground_truth, args.patient_id)
        print("\n" + "=" * 60)
        print("MASK CHECK COMPLETED")
        print("=" * 60)
    except Exception as exc:
        print("ERROR: Fatal mask validation failure.")
        print(str(exc))
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()