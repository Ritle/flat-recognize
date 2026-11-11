"""Quarantine audited training failures without altering evaluation lists."""

import argparse
import hashlib
import json
from pathlib import Path


def prepare(report_path, source_splits, output):
    report_path, source_splits, output = map(Path, (report_path, source_splits, output))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    dataset = Path(report["settings"]["dataset"])
    if report.get("substituted_samples") != 0:
        raise ValueError("Expected a strict audit without substituted samples")
    rows = {(r["split"], r["sample"]): r for r in report["samples"]}
    if len(rows) != report["total"]:
        raise ValueError("Incomplete or repeated audit records")
    selected, texts, source_hashes = {}, {}, {}
    for split in ("train", "val", "test"):
        original = dataset / f"{split}.txt"
        if hashlib.sha256(original.read_bytes()).hexdigest() != report["source_split_sha256"][split]:
            raise ValueError("Official source split changed after SVG audit")
        file = source_splits / f"{split}.txt"
        source_hashes[split] = hashlib.sha256(file.read_bytes()).hexdigest()
        texts[split] = file.read_text(encoding="utf-8")
        selected[split] = [line.strip().strip("/") for line in texts[split].splitlines() if line.strip()]
        if len(selected[split]) != len(set(selected[split])):
            raise ValueError("Repeated selected sample")
        for sample in selected[split]:
            if (split, sample) not in rows:
                raise ValueError("Selected sample is absent from audited split")
            path = (dataset / sample / "model.svg").resolve()
            if not path.is_relative_to(dataset.resolve()):
                raise ValueError("Unsafe sample path")
            if hashlib.sha256(path.read_bytes()).hexdigest() != rows[split, sample]["svg_sha256"]:
                raise ValueError(f"SVG changed after audit: {split}/{sample}")
    invalid = {split: [sample for sample in selected[split] if rows[split, sample]["status"] != "ok"]
               for split in selected}
    if invalid["val"]:
        raise ValueError("Validation contains SVG failures; resolve them before preparing a training run")
    kept = [sample for sample in selected["train"] if rows["train", sample]["status"] == "ok"]
    if not kept:
        raise ValueError("No valid training samples remain")
    output.mkdir(parents=True, exist_ok=False)
    (output / "train.txt").write_text("\n".join(f"/{sample}/" for sample in kept) + "\n", encoding="utf-8")
    for split in ("val", "test"):
        (output / f"{split}.txt").write_bytes((source_splits / f"{split}.txt").read_bytes())
    result = {"version": 1, "audit_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
              "source_split_sha256": source_hashes, "excluded_train_samples": invalid["train"],
              "known_test_failures_retained": invalid["test"],
              "counts": {"train": len(kept), "val": len(selected["val"]), "test": len(selected["test"])},
              "evaluation_lists_unchanged": True}
    (output / "provenance.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--source-splits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.audit, args.source_splits, args.output), indent=2))


if __name__ == "__main__":
    main()
