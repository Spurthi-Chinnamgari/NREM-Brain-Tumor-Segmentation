import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
POSTPROCESSING_ROOT = PROJECT_ROOT / "postprocessing"
RESULTS_DIR = POSTPROCESSING_ROOT / "results"
INPUT_MANIFEST = POSTPROCESSING_ROOT / "input" / "inputs.json"
METRICS_PATH = RESULTS_DIR / "metrics.json"
COMPARISON_PATH = RESULTS_DIR / "metric_comparison.json"
FEATURES_PATH = RESULTS_DIR / "features.json"
FEATURE_REPORT_PATH = RESULTS_DIR / "feature_report.json"
REPORT_PATH = RESULTS_DIR / "final_report.txt"

REGIONS = ("WT", "TC", "ET")
METRICS = ("Dice", "IoU", "Precision", "Recall", "HD95", "ASD")


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required report input does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _source_paths(metrics: dict, inputs: dict | None) -> dict[str, str | None]:
    paths = {
        "prediction": metrics.get("prediction_path"),
        "ground_truth": metrics.get("ground_truth_path"),
        "postprocessed": metrics.get("postprocessed_prediction_path"),
    }
    if inputs:
        paths["prediction"] = inputs.get("prediction_path", paths["prediction"])
        paths["ground_truth"] = inputs.get("ground_truth_path", paths["ground_truth"])
    return paths


def _geometry_status(paths: dict[str, str | None]) -> tuple[dict[str, object], list[str]]:
    status: dict[str, object] = {
        "prediction_exists": False,
        "ground_truth_exists": False,
        "shape": None,
        "voxel_spacing": None,
        "shape_compatible": False,
        "spacing_compatible": False,
        "affine_compatible": False,
    }
    problems = []
    prediction_path = Path(paths["prediction"]).expanduser() if paths["prediction"] else None
    ground_truth_path = Path(paths["ground_truth"]).expanduser() if paths["ground_truth"] else None

    if prediction_path is None or not prediction_path.is_file():
        problems.append("prediction artifact is missing")
    else:
        status["prediction_exists"] = True
    if ground_truth_path is None or not ground_truth_path.is_file():
        problems.append("ground-truth artifact is missing")
    else:
        status["ground_truth_exists"] = True

    if not status["prediction_exists"] or not status["ground_truth_exists"]:
        return status, problems

    prediction = nib.load(str(prediction_path))
    ground_truth = nib.load(str(ground_truth_path))
    status["shape"] = list(prediction.shape)
    status["voxel_spacing"] = [float(value) for value in prediction.header.get_zooms()[:3]]
    status["shape_compatible"] = prediction.shape == ground_truth.shape
    status["spacing_compatible"] = bool(
        np.allclose(
            prediction.header.get_zooms()[:3],
            ground_truth.header.get_zooms()[:3],
            rtol=1e-5,
            atol=1e-6,
        )
    )
    status["affine_compatible"] = bool(np.allclose(prediction.affine, ground_truth.affine, rtol=1e-5, atol=1e-6))
    if not status["shape_compatible"]:
        problems.append("prediction and ground truth shapes differ")
    if not status["spacing_compatible"]:
        problems.append("prediction and ground truth spacing differs")
    if not status["affine_compatible"]:
        problems.append("prediction and ground truth affines differ")
    return status, problems


def _existing_explainability_paths(patient_id: str) -> dict[str, list[str]]:
    gradcam_paths = sorted(
        str(path.resolve())
        for path in (POSTPROCESSING_ROOT / "gradcam_results").glob(f"{patient_id}*")
        if path.is_file()
    )
    segmentation_paths = []
    for directory in (POSTPROCESSING_ROOT / "results", POSTPROCESSING_ROOT / "postprocessed"):
        if directory.is_dir():
            segmentation_paths.extend(
                str(path.resolve())
                for path in directory.glob(f"{patient_id}*")
                if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
            )
    reliability_paths = sorted(
        str(path.resolve())
        for path in POSTPROCESSING_ROOT.rglob("*")
        if path.is_file() and any(token in path.name.lower() for token in ("mrem", "reliability"))
    )
    return {
        "segmentation_visualization": sorted(set(segmentation_paths)),
        "gradcam_explanation": gradcam_paths,
        "reliability_mrem_visualization": reliability_paths,
    }


def _format_metrics(comparison: dict) -> list[str]:
    lines = ["SEGMENTATION PERFORMANCE", "-------------------------"]
    for region in REGIONS:
        lines.append(region)
        region_values = comparison["comparison"].get(region, {})
        for metric in METRICS:
            values = region_values.get(metric, {})
            lines.append(
                f"  {metric}: before={values.get('before')} | after={values.get('after')} "
                f"| delta={values.get('delta_after_minus_before')}"
            )
    return lines


def _format_features(features: dict) -> list[str]:
    lines = ["TUMOR FEATURES (AI/MODEL OUTPUT)", "-------------------------------"]
    for region in REGIONS:
        values = features.get("regions", {}).get(region)
        lines.append(region)
        if not values:
            lines.append("  Source data unavailable")
            continue
        lines.extend([
            f"  Tumor present: {values.get('tumor_present')}",
            f"  Volume: {values.get('volume_mm3')} mm3 ({values.get('volume_cm3')} cm3)",
            f"  Voxel count: {values.get('voxel_count')}",
            f"  Centroid voxel: {values.get('centroid_voxel')}",
            f"  Centroid world mm: {values.get('centroid_world_mm')}",
            f"  Location: {values.get('location')}",
            f"  Bounding box mm: {values.get('bounding_box_mm')}",
            f"  Elongation: {values.get('elongation')}",
            f"  Sphericity: {values.get('sphericity')}",
            f"  Compactness: {values.get('compactness')}",
            f"  Surface area mm2: {values.get('surface_area_mm2')}",
        ])
    return lines


def build_report(patient_id: str) -> str:
    metrics = _load_json(METRICS_PATH)
    inputs = _load_json(INPUT_MANIFEST) if INPUT_MANIFEST.is_file() else None
    comparison = _load_json(COMPARISON_PATH) if COMPARISON_PATH.is_file() else {
        "comparison": {
            region: {
                metric: {
                    "before": metrics["metrics"]["before"][region].get(metric),
                    "after": metrics["metrics"]["after"][region].get(metric),
                    "delta_after_minus_before": None,
                }
                for metric in METRICS
            }
            for region in REGIONS
        }
    }
    features = _load_json(FEATURE_REPORT_PATH if FEATURE_REPORT_PATH.is_file() else FEATURES_PATH)

    source_patient_id = metrics.get("patient_id") or (inputs or {}).get("patient_id")
    if source_patient_id != patient_id:
        raise ValueError(f"Requested patient {patient_id!r} does not match report artifacts for {source_patient_id!r}")

    paths = _source_paths(metrics, inputs)
    validation, problems = _geometry_status(paths)
    validation_status = "PASS" if not problems else "INCOMPLETE: " + "; ".join(problems)
    explainability = _existing_explainability_paths(patient_id)

    lines = [
        "POSTPROCESSING TECHNICAL REPORT",
        "===============================",
        "This report describes AI/model segmentation outputs for technical evaluation.",
        "It is not a patient-facing medical finding or diagnosis.",
        "",
        "PATIENT AND INPUTS",
        "------------------",
        f"Patient ID: {patient_id}",
        f"Prediction: {paths['prediction'] or 'Not available'}",
        f"Ground truth: {paths['ground_truth'] or 'Not available'}",
        f"Postprocessed prediction: {paths['postprocessed'] or 'Not available'}",
        f"Image shape: {validation['shape']}",
        f"Voxel spacing mm: {validation['voxel_spacing']}",
        f"Validation status: {validation_status}",
        "",
        "POSTPROCESSING SUMMARY",
        "----------------------",
        "Original and cleaned prediction artifacts are referenced by the existing metrics output.",
        f"Cleaned prediction exists: {Path(paths['postprocessed']).is_file() if paths['postprocessed'] else False}",
        "No additional morphology values were calculated by this report layer.",
        "",
    ]
    lines.extend(_format_metrics(comparison))
    lines.extend(["", ""])
    lines.extend(_format_features(features))
    lines.extend(["", "EXPLAINABILITY AND VISUALIZATION", "--------------------------------"])
    for name, label in (
        ("segmentation_visualization", "Segmentation visualization"),
        ("gradcam_explanation", "Grad-CAM explanation"),
        ("reliability_mrem_visualization", "Reliability/MREM visualization"),
    ):
        found = explainability[name]
        if found:
            lines.append(f"{label}:")
            lines.extend(f"  {path}" for path in found)
        else:
            lines.append(f"{label}: No existing artifact found")
    lines.extend([
        "",
        "SOURCE REPORT FILES",
        "-------------------",
        f"Metrics JSON: {METRICS_PATH.resolve()}",
        f"Features JSON: {(FEATURE_REPORT_PATH if FEATURE_REPORT_PATH.is_file() else FEATURES_PATH).resolve()}",
        f"Metric comparison JSON: {COMPARISON_PATH.resolve() if COMPARISON_PATH.is_file() else 'Not available'}",
        f"Input manifest: {INPUT_MANIFEST.resolve() if INPUT_MANIFEST.is_file() else 'Not available'}",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble a generic technical postprocessing report from existing artifacts.")
    parser.add_argument("--patient-id", required=True)
    args = parser.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(build_report(args.patient_id), encoding="utf-8")
    print(f"Saved final report: {REPORT_PATH.resolve()}")


if __name__ == "__main__":
    main()