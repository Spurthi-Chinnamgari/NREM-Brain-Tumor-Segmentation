import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
POSTPROCESSING_DIR = PROJECT_ROOT / "postprocessing"
RESULTS_DIR = POSTPROCESSING_DIR / "results"
FEATURES_PATH = RESULTS_DIR / "features.json"
FEATURE_REPORT_PATH = RESULTS_DIR / "feature_report.json"
FINAL_REPORT_PATH = RESULTS_DIR / "final_report.txt"


def _run(script: str, *arguments: str) -> None:
    command = [sys.executable, str(POSTPROCESSING_DIR / script), *arguments]
    print(f"\n>>> {' '.join(command)}")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def _write_feature_report() -> None:
    with FEATURES_PATH.open("r", encoding="utf-8") as handle:
        features = json.load(handle)
    with FEATURE_REPORT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(features, handle, indent=2, sort_keys=True, allow_nan=False)


def _write_final_report(patient_id: str) -> None:
    with (RESULTS_DIR / "metrics.json").open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    with (RESULTS_DIR / "metric_comparison.json").open("r", encoding="utf-8") as handle:
        comparison = json.load(handle)
    with FEATURES_PATH.open("r", encoding="utf-8") as handle:
        features = json.load(handle)

    lines = [
        "POSTPROCESSING FINAL REPORT",
        "===========================",
        f"Patient: {patient_id}",
        "",
        "IMPORTANT OUTPUTS",
        f"Original prediction: {metrics['prediction_path']}",
        f"Cleaned prediction: {metrics['postprocessed_prediction_path']}",
        f"Ground truth: {metrics['ground_truth_path']}",
        f"Metrics: {RESULTS_DIR / 'metrics.json'}",
        f"Metric comparison: {RESULTS_DIR / 'metric_comparison.json'}",
        f"Features: {FEATURES_PATH}",
        f"Feature report: {FEATURE_REPORT_PATH}",
        "",
        "METRIC COMPARISON (BEFORE -> AFTER)",
    ]
    for region in ("WT", "TC", "ET"):
        lines.append(region)
        for metric, values in comparison["comparison"][region].items():
            lines.append(f"  {metric}: {values['before']} -> {values['after']} (delta {values['delta_after_minus_before']})")

    lines.extend(["", "EXTRACTED FEATURES"])
    for region in ("WT", "TC", "ET"):
        result = features["regions"][region]
        lines.extend([
            region,
            f"  Tumor present: {result['tumor_present']}",
            f"  Volume: {result['volume_mm3']} mm3 ({result['volume_cm3']} cm3)",
            f"  Bounding box (mm): {result['bounding_box_mm']}",
            f"  Centroid voxel: {result['centroid_voxel']}",
            f"  Centroid world (mm): {result['centroid_world_mm']}",
            f"  Location: {result['location']}",
            f"  Elongation: {result['elongation']}",
            f"  Sphericity: {result['sphericity']}",
            f"  Compactness: {result['compactness']}",
            f"  Surface area (mm2): {result['surface_area_mm2']}",
        ])

    FINAL_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete existing postprocessing pipeline.")
    parser.add_argument("--patient-id", required=True)
    args = parser.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    _run("prepare_inputs.py", "--patient-id", args.patient_id)
    _run("postprocess.py", "--patient-id", args.patient_id)
    _run("metrics.py", "--patient-id", args.patient_id)
    _run("feature_extraction.py")
    _run("compare_metrics.py")
    _write_feature_report()
    _write_final_report(args.patient_id)
    print(f"\nSaved feature report: {FEATURE_REPORT_PATH}")
    print(f"Saved final report: {FINAL_REPORT_PATH}")


if __name__ == "__main__":
    main()