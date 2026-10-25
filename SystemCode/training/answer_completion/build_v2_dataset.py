"""Responsibilities: combine approved supplemental data with V1 training rows without leakage.

Implementation: retain only V1 training rows, preserve all V2 groups and splits, prefix provenance IDs,
check exact Unicode uniqueness, and copy the new challenge and known regression into private staging.
Related Modules: annotate produces blind teacher labels; train_v2 reads the resulting fixed files.

Declaration Index:
- read_jsonl: read every source record without changing labels or text.
- checksum: hash persisted bytes for input provenance.
- main: assemble V2 data, enforce source/split contracts, and write reproducible staging metadata.

Variable Index:
None

Constraints:
Old validation/test rows are never added to training; there is no resampling, implicit repair,
relabeling, parameter change, or real-user record. Source data and original IDs remain recoverable."""

import argparse
import hashlib
import json
import shutil
import unicodedata
from collections import Counter
from pathlib import Path


def read_jsonl(path):
    """Inputs: UTF-8 JSONL path. Outputs: ordered rows; filesystem/JSON errors propagate.
    Logic: retain source text/labels and attached blind annotation provenance without filtering.
    """
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def checksum(path):
    """Inputs: readable file path. Outputs: SHA256 hex string of actual persisted source bytes.
    Logic: source integrity is recorded before any experiment consumes the assembled data.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    """Inputs: explicit V1/V2 audited directories and destination. Outputs: fixed combined dataset.
    Logic: prefix IDs/groups by experiment, retain V1 training only, check duplicate and group-split
    contracts, then save source hashes/config/challenge snapshots. Invalid input aborts explicitly.
    Constraints: source files are preserved, no old held-out rows or known challenge train examples.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1", type=Path, required=True)
    parser.add_argument("--v2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    sources = {"v1": args.v1 / "data.jsonl", "v2": args.v2 / "data.jsonl"}
    records = []
    for version, path in sources.items():
        for item in read_jsonl(path):
            if version == "v1" and item["split"] != "train":
                continue
            records.append({**item, "original_id": item["id"], "source_experiment": version,
                            "id": version + "/" + item["id"],
                            "group_id": version + "/" + item["group_id"]})
    expected = Counter({"train": 2400, "validation": 200, "test": 200})
    if Counter(x["split"] for x in records) != expected:
        raise ValueError("Combined dataset violates the approved source/split counts")
    normalized = [unicodedata.normalize("NFKC", x["text"]).casefold() for x in records]
    if len(set(normalized)) != len(normalized):
        raise ValueError("Combined data has duplicates; preserve source and explicitly review")
    groups = {}
    for row in records:
        if groups.setdefault(row["group_id"], row["split"]) != row["split"]:
            raise ValueError("Source group crosses split boundary")
    args.output.mkdir(parents=True, exist_ok=False)
    data_path = args.output / "data.jsonl"
    data_path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in records),
                         encoding="utf-8")
    config_path = here / "experiment-v2.json"
    shutil.copy2(config_path, args.output / "experiment.json")
    shutil.copy2(here / "challenge-v2.jsonl", args.output / "challenge.jsonl")
    shutil.copy2(here / "challenge.jsonl", args.output / "regression-v1.jsonl")
    provenance = {
        "source_data_sha256": {k: checksum(p) for k, p in sources.items()},
        "source_provenance": {k: json.loads((p.parent / "provenance.json").read_text(
            encoding="utf-8")) for k, p in sources.items()},
        "data_sha256": checksum(data_path), "records": len(records),
        "split_counts": dict(Counter(x["split"] for x in records)),
        "v1_use": "training rows only; V1 validation/test never become training rows",
        "challenge_role": "new assistant-authored semantic holdout, not human gold",
        "regression_role": "known V1 challenge, reported separately and excluded from acceptance",
        "challenge_sha256": checksum(args.output / "challenge.jsonl"),
        "regression_sha256": checksum(args.output / "regression-v1.jsonl"),
        "public_training_data": "None", "real_user_data": "None",
        "class_counts": dict(Counter(f'{x["split"]}/{x["language"]}/{x["label"]}'
                                     for x in records)),
    }
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print("assembled", len(records), provenance["class_counts"], flush=True)


if __name__ == "__main__":
    main()
