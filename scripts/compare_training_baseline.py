"""Screen a training candidate on unchanged validation cohorts; never deploy it."""

import argparse
import json
import math
from pathlib import Path

CLASSES = ("floor", "wall", "door", "window")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def compare(baseline, candidate, rules):
    for key in ("max_base_mean_iou_drop", "max_base_structural_class_iou_drop"):
        value = rules.get(key)
        if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Invalid screening threshold: {key}")
    for key in ("new_wall_must_not_drop", "new_door_and_window_must_improve"):
        if type(rules.get(key)) is not bool:
            raise ValueError(f"Invalid screening condition: {key}")
    deltas = {}
    for source in ("svg", "raster"):
        before, after = baseline[source], candidate[source]
        for row in (before, after):
            if (row["split"] != "val" or row["source"] != source or
                    row["limit_per_source"] is not None or row["sample_count"] <= 0):
                raise ValueError("Expected complete source-specific validation reports")
            values = list(row["iou"].values())
            if (set(row["iou"]) != set(CLASSES) or
                    not all(math.isfinite(value) and 0 <= value <= 1 for value in values) or
                    not math.isclose(row["miou"], sum(values) / 4, abs_tol=1e-7)):
                raise ValueError("Invalid IoU metrics")
        for key in ("sources", "sample_count", "evaluated_pixels", "ignored_pixels", "device"):
            if before[key] != after[key]:
                raise ValueError(f"Evaluation cohort or protocol differs: {source}/{key}")
        for key in ("image_size", "normalize", "letterbox"):
            if before["config"]["data"].get(key) != after["config"]["data"].get(key):
                raise ValueError(f"Input preprocessing differs: {key}")
        if (source == "svg" and before["config"]["data"].get("svg_input_mode", "structural") !=
                after["config"]["data"].get("svg_input_mode", "structural")):
            raise ValueError("SVG input mode differs")
        if before["config"]["model"]["encoder_name"] != after["config"]["model"]["encoder_name"]:
            raise ValueError("Model architecture differs")
        if before["config"]["optim"]["batch_size"] != after["config"]["optim"]["batch_size"]:
            raise ValueError("Evaluation batch size differs")
        deltas[source] = {"miou": after["miou"] - before["miou"],
                          "iou": {name: after["iou"][name] - before["iou"][name] for name in CLASSES}}
    for reports in (baseline, candidate):
        if reports["svg"]["checkpoint_sha256"] != reports["raster"]["checkpoint_sha256"]:
            raise ValueError("Source reports use different checkpoints")
    failed = []
    if deltas["svg"]["miou"] < -rules["max_base_mean_iou_drop"]:
        failed.append("base_mean_iou_regression")
    for name in ("wall", "door", "window"):
        if deltas["svg"]["iou"][name] < -rules["max_base_structural_class_iou_drop"]:
            failed.append(f"base_{name}_iou_regression")
    if rules["new_wall_must_not_drop"] and deltas["raster"]["iou"]["wall"] < 0:
        failed.append("new_wall_iou_regression")
    if rules["new_door_and_window_must_improve"]:
        for name in ("door", "window"):
            if deltas["raster"]["iou"][name] <= 0:
                failed.append(f"new_{name}_iou_did_not_improve")
    return {"version": 1, "rules": rules, "deltas": deltas, "failed_checks": failed,
            "eligible_for_pipeline_check": not failed, "deploy": False,
            "new_validation_samples": candidate["raster"]["sample_count"],
            "baseline_checkpoint_sha256": baseline["svg"]["checkpoint_sha256"],
            "candidate_checkpoint_sha256": candidate["svg"]["checkpoint_sha256"],
            "test_used": False}


def compare_with_clean_control(baseline, candidate, clean_before, clean_after, rules, clean_max_drop):
    """Require improvement screening and preserved quality on structural inputs."""
    for reports in (baseline, candidate):
        if reports["svg"]["config"]["data"].get("svg_input_mode", "structural") != "full":
            raise ValueError("Full-scene protocol requires full-scene primary reports")
    for report in (clean_before, clean_after):
        if report["config"]["data"].get("svg_input_mode", "structural") != "structural":
            raise ValueError("Clean control requires structural input reports")
    result = compare(baseline, candidate, rules)
    control_rules = {**rules, "max_base_mean_iou_drop": clean_max_drop,
                     "new_wall_must_not_drop": False, "new_door_and_window_must_improve": False}
    # Reusing the raster reports also verifies that both views use the same
    # original/candidate checkpoint, rather than borrowing a previous clean score.
    control = compare({"svg": clean_before, "raster": baseline["raster"]},
                      {"svg": clean_after, "raster": candidate["raster"]}, control_rules)
    result["clean_control"] = control
    result["failed_checks"] += [f"clean_control_{reason}" for reason in control["failed_checks"]]
    result["eligible_for_pipeline_check"] = not result["failed_checks"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--clean-control", type=Path,
                        help="candidate structural val-svg.json; required by a full-scene protocol")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    before = {source: read(args.baseline / f"val-{source}.json") for source in ("svg", "raster")}
    after = {source: read(args.candidate / f"val-{source}.json") for source in ("svg", "raster")}
    protocol = read(args.protocol)
    if before["svg"]["checkpoint_sha256"] != protocol["initial_checkpoint_sha256"]:
        raise ValueError("Baseline checkpoint differs from the pilot protocol")
    clean_baseline = protocol.get("clean_control_baseline")
    uses_full = any(reports["svg"]["config"]["data"].get("svg_input_mode", "structural") == "full"
                    for reports in (before, after))
    if uses_full and not clean_baseline:
        raise ValueError("Full-scene protocol must define clean_control_baseline")
    if clean_baseline:
        if args.clean_control is None:
            raise ValueError("Full-scene protocol requires --clean-control; comparison was not saved")
        control_before = read(Path(clean_baseline).parent / "val-svg.json")
        control_after = read(args.clean_control)
        result = compare_with_clean_control(before, after, control_before, control_after,
                                            protocol["screening_rules"],
                                            protocol["clean_control_max_mean_iou_drop"])
        result["clean_control"].update(baseline=str(Path(clean_baseline).parent / "val-svg.json"),
                                       candidate=str(args.clean_control))
    else:
        if args.clean_control is not None:
            raise ValueError("Protocol does not define a clean-control baseline")
        result = compare(before, after, protocol["screening_rules"])
    result.update(baseline=str(args.baseline), candidate=str(args.candidate), protocol=str(args.protocol))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))
    return 0 if result["eligible_for_pipeline_check"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
