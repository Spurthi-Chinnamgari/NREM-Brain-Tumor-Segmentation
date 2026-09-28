import argparse
import json
from pathlib import Path

try:
    from .check_masks import validate_prediction_ground_truth
except ImportError:  # pragma: no cover - direct script execution
    from check_masks import validate_prediction_ground_truth


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "postprocessing" / "input"
MANIFEST_PATH = INPUT_DIR / "inputs.json"


def default_prediction_path(patient_id: str) -> Path:
    return PROJECT_ROOT / "Attention_unet" / "results" / "predictions" / f"{patient_id}_prediction_original.nii.gz"


def default_ground_truth_path(patient_id: str) -> Path:
    return PROJECT_ROOT / "data" / "brats2023" / patient_id / f"{patient_id}-seg.nii.gz"


def prepare_inputs(patient_id: str) -> dict[str, str]:
    prediction_path = default_prediction_path(patient_id).resolve()
    ground_truth_path = default_ground_truth_path(patient_id).resolve()
    validate_prediction_ground_truth(prediction_path, ground_truth_path, patient_id)

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "patient_id": patient_id,
        "prediction_path": str(prediction_path),
        "ground_truth_path": str(ground_truth_path),
    }
    with MANIFEST_PATH.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Register the real inference prediction and BraTS ground truth for postprocessing.")
    parser.add_argument("--patient-id", required=True)
    args = parser.parse_args()
    manifest = prepare_inputs(args.patient_id)
    print(f"Prepared postprocessing input manifest: {MANIFEST_PATH}")
    print(f"Prediction: {manifest['prediction_path']}")
    print(f"Ground truth: {manifest['ground_truth_path']}")


if __name__ == "__main__":
    main()