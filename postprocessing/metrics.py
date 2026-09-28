import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

from check_masks import validate_prediction_ground_truth

PROJECT_ROOT = Path(__file__).resolve().parents[1]
POSTPROCESSING_ROOT = PROJECT_ROOT / "postprocessing"
RESULTS_DIR = POSTPROCESSING_ROOT / "results"

REGIONS = {
    "WT": [1, 2, 3],
    "TC": [1, 3],
    "ET": [3],
}


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


def _default_postprocessed_prediction_path(patient_id: str | None = None) -> Path:
    if patient_id is None:
        patient_id = _default_prediction_path().stem.replace("_prediction_original", "").replace("_prediction_original.nii", "")
    return POSTPROCESSING_ROOT / "postprocessed" / "cleaned_prediction_mask.nii.gz"


def _default_ground_truth_path(patient_id: str | None = None) -> Path:
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


def _load_label_volume(path: str | Path, label_name: str) -> tuple[np.ndarray, nib.Nifti1Image]:
    image_path = Path(path).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"{label_name} mask does not exist: {image_path}")
    image = nib.load(str(image_path))
    volume = np.asarray(image.get_fdata(dtype=np.float64), dtype=np.float64)
    if volume.ndim != 3:
        raise ValueError(f"{label_name} mask must be 3D: {image_path}")
    if not np.all(np.isfinite(volume)):
        raise ValueError(f"{label_name} mask contains NaN/Inf values: {image_path}")
    rounded = np.rint(volume)
    if not np.allclose(volume, rounded, rtol=0.0, atol=1e-6):
        raise ValueError(f"{label_name} mask contains non-integer labels: {image_path}")
    integer_volume = rounded.astype(np.int16)
    return integer_volume, image


def _validate_pair(prediction_path: str | Path, ground_truth_path: str | Path, label_name: str) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    prediction, prediction_nii = _load_label_volume(prediction_path, f"{label_name} prediction")
    ground_truth, ground_truth_nii = _load_label_volume(ground_truth_path, "Ground truth")

    if prediction.shape != ground_truth.shape:
        raise ValueError(
            f"{label_name} prediction and ground truth have mismatched shapes: "
            f"prediction={prediction.shape}, ground_truth={ground_truth.shape}."
        )

    prediction_spacing = tuple(float(v) for v in prediction_nii.header.get_zooms()[:3])
    ground_truth_spacing = tuple(float(v) for v in ground_truth_nii.header.get_zooms()[:3])
    if not np.allclose(prediction_spacing, ground_truth_spacing, rtol=1e-5, atol=1e-6):
        raise ValueError(
            f"{label_name} prediction and ground truth voxel spacing are incompatible: "
            f"prediction={prediction_spacing}, ground_truth={ground_truth_spacing}."
        )

    if not np.allclose(prediction_nii.affine, ground_truth_nii.affine, rtol=1e-5, atol=1e-6):
        raise ValueError(
            f"{label_name} prediction and ground truth affine matrices are incompatible: "
            f"prediction_affine={prediction_nii.affine}, ground_truth_affine={ground_truth_nii.affine}."
        )

    return prediction, ground_truth, prediction_spacing


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    pred_sum = int(np.count_nonzero(pred))
    gt_sum = int(np.count_nonzero(gt))
    if pred_sum == 0 and gt_sum == 0:
        return 1.0
    if pred_sum == 0 or gt_sum == 0:
        return 0.0
    intersection = int(np.count_nonzero(pred & gt))
    return float((2.0 * intersection) / (pred_sum + gt_sum))


def iou_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    intersection = int(np.count_nonzero(pred & gt))
    union = int(np.count_nonzero(pred | gt))
    if union == 0:
        return 1.0
    return float(intersection / union)


def precision_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    true_positive = int(np.count_nonzero(pred & gt))
    false_positive = int(np.count_nonzero(pred & ~gt))
    denominator = true_positive + false_positive
    if denominator == 0:
        return 1.0 if np.count_nonzero(gt) == 0 else 0.0
    return float(true_positive / denominator)


def recall_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    true_positive = int(np.count_nonzero(pred & gt))
    false_negative = int(np.count_nonzero((~pred) & gt))
    denominator = true_positive + false_negative
    if denominator == 0:
        return 1.0 if np.count_nonzero(pred) == 0 else 0.0
    return float(true_positive / denominator)


def surface_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return mask.copy()
    structure = ndimage.generate_binary_structure(3, 1)
    eroded = ndimage.binary_erosion(mask, structure=structure, border_value=0)
    return mask ^ eroded


def hd95(pred: np.ndarray, gt: np.ndarray, spacing: tuple[float, float, float]) -> float:
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    if not np.any(pred) and not np.any(gt):
        return 0.0
    if not np.any(pred) or not np.any(gt):
        return float("nan")

    pred_surface = surface_mask(pred)
    gt_surface = surface_mask(gt)
    pred_distance = ndimage.distance_transform_edt(~pred_surface, sampling=spacing)
    gt_distance = ndimage.distance_transform_edt(~gt_surface, sampling=spacing)
    all_distances = np.concatenate([
        pred_distance[gt_surface],
        gt_distance[pred_surface],
    ])
    return float(np.percentile(all_distances, 95))


def asd(pred: np.ndarray, gt: np.ndarray, spacing: tuple[float, float, float]) -> float:
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    if not np.any(pred) and not np.any(gt):
        return 0.0
    if not np.any(pred) or not np.any(gt):
        return float("nan")

    pred_surface = surface_mask(pred)
    gt_surface = surface_mask(gt)
    pred_distance = ndimage.distance_transform_edt(~pred_surface, sampling=spacing)
    gt_distance = ndimage.distance_transform_edt(~gt_surface, sampling=spacing)
    all_distances = np.concatenate([
        pred_distance[gt_surface],
        gt_distance[pred_surface],
    ])
    return float(np.mean(all_distances))


def _compute_region_metrics(pred: np.ndarray, gt: np.ndarray, spacing: tuple[float, float, float], region_name: str, labels: list[int]) -> dict[str, float | str | None]:
    pred_region = np.isin(pred, labels)
    gt_region = np.isin(gt, labels)

    metrics = {
        "Dice": dice_score(pred_region, gt_region),
        "IoU": iou_score(pred_region, gt_region),
        "Precision": precision_score(pred_region, gt_region),
        "Recall": recall_score(pred_region, gt_region),
        "HD95": hd95(pred_region, gt_region, spacing),
        "ASD": asd(pred_region, gt_region, spacing),
    }
    return {"region": region_name, **metrics}


def _format_value(value: float | str | None) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float) and np.isnan(value):
        return "N/A"
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def _summarize_metrics(results: dict[str, dict[str, float | str | None]]) -> dict[str, dict[str, float | str | None]]:
    summary = {}
    for region_name, metrics in results.items():
        summary[region_name] = {
            "Dice": float(metrics["Dice"]),
            "IoU": float(metrics["IoU"]),
            "Precision": float(metrics["Precision"]),
            "Recall": float(metrics["Recall"]),
            "HD95": metrics["HD95"],
            "ASD": metrics["ASD"],
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare the original prediction and postprocessed prediction against ground truth for BraTS regions.")
    parser.add_argument("--prediction", type=str, default=None, help="Path to the original reconstructed prediction NIfTI.")
    parser.add_argument("--postprocessed", type=str, default=None, help="Path to the cleaned postprocessed prediction NIfTI.")
    parser.add_argument("--ground-truth", type=str, default=None, help="Path to the raw BraTS ground-truth NIfTI.")
    parser.add_argument("--patient-id", type=str, default=None, help="Patient ID used if default paths are needed.")
    args = parser.parse_args()

    patient_id = args.patient_id
    prediction_path = Path(args.prediction) if args.prediction else _default_prediction_path(patient_id)
    postprocessed_path = Path(args.postprocessed) if args.postprocessed else _default_postprocessed_prediction_path(patient_id)
    ground_truth_path = Path(args.ground_truth) if args.ground_truth else _default_ground_truth_path(patient_id)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        original, ground_truth, spacing = _validate_pair(prediction_path, ground_truth_path, "Original")
        postprocessed, _, _ = _validate_pair(postprocessed_path, ground_truth_path, "Postprocessed")
    except Exception as exc:
        print("ERROR: Prediction/ground-truth compatibility check failed before metric computation.")
        print(str(exc))
        raise SystemExit(1) from exc

    metric_sets = {"before": {}, "after": {}}
    for region_name, labels in REGIONS.items():
        metric_sets["before"][region_name] = _compute_region_metrics(original, ground_truth, spacing, region_name, labels)
        metric_sets["after"][region_name] = _compute_region_metrics(postprocessed, ground_truth, spacing, region_name, labels)

    print("=" * 80)
    print("BRAINS TUMOR SEGMENTATION METRICS")
    print("=" * 80)
    print(f"Patient ID: {patient_id or prediction_path.name.replace('_prediction_original.nii.gz', '')}")
    print(f"Original prediction: {prediction_path}")
    print(f"Postprocessed prediction: {postprocessed_path}")
    print(f"Ground truth: {ground_truth_path}")

    for label in ("before", "after"):
        print(f"\n[{label.upper()}]")
        for region_name in ["WT", "TC", "ET"]:
            region_metrics = metric_sets[label][region_name]
            print(f"\n{region_name}")
            print("-" * 30)
            for metric_name in ["Dice", "IoU", "Precision", "Recall", "HD95", "ASD"]:
                value = region_metrics[metric_name]
                print(f"{metric_name}: {_format_value(value)}")

    result_payload = {
        "patient_id": patient_id or prediction_path.name.replace("_prediction_original.nii.gz", ""),
        "prediction_path": str(prediction_path),
        "postprocessed_prediction_path": str(postprocessed_path),
        "ground_truth_path": str(ground_truth_path),
        "metrics": {
            "before": _summarize_metrics(metric_sets["before"]),
            "after": _summarize_metrics(metric_sets["after"]),
        },
    }

    output_json = RESULTS_DIR / "metrics.json"
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(result_payload, handle, indent=2)

    print(f"\nSaved metric results to: {output_json}")
    print("=" * 80)


if __name__ == "__main__":
    main()