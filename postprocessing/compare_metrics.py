import argparse
import json
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "postprocessing" / "results" / "metrics.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "postprocessing" / "results" / "metric_comparison.json"
REGIONS = ("WT", "TC", "ET")
METRICS = ("Dice", "IoU", "Precision", "Recall", "HD95", "ASD")


def _finite_or_none(value: object) -> float | None:
    if value is None:
        return None
    numeric_value = float(value)
    return numeric_value if math.isfinite(numeric_value) else None


def compare_metrics(input_path: str | Path = DEFAULT_INPUT, output_path: str | Path = DEFAULT_OUTPUT) -> dict:
    source_path = Path(input_path).expanduser().resolve()
    with source_path.open("r", encoding="utf-8") as handle:
        source = json.load(handle)

    metric_sets = source.get("metrics", {})
    if set(metric_sets) != {"before", "after"}:
        raise ValueError(f"Expected before/after metrics in {source_path}")

    comparison = {}
    for region in REGIONS:
        if region not in metric_sets["before"] or region not in metric_sets["after"]:
            raise ValueError(f"Missing {region} metrics in {source_path}")
        comparison[region] = {}
        for metric in METRICS:
            before = _finite_or_none(metric_sets["before"][region].get(metric))
            after = _finite_or_none(metric_sets["after"][region].get(metric))
            delta = after - before if before is not None and after is not None else None
            comparison[region][metric] = {
                "before": before,
                "after": after,
                "delta_after_minus_before": delta,
            }

    result = {
        "patient_id": source.get("patient_id"),
        "source_metrics_path": str(source_path),
        "comparison": comparison,
    }
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare existing before/after postprocessing metrics.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = compare_metrics(args.input, args.output)

    print("METRIC COMPARISON")
    for region in REGIONS:
        print(region)
        for metric in METRICS:
            values = result["comparison"][region][metric]
            print(f"  {metric}: before={values['before']} after={values['after']} delta={values['delta_after_minus_before']}")
    print(f"Saved comparison: {Path(args.output).expanduser().resolve()}")


if __name__ == "__main__":
    main()