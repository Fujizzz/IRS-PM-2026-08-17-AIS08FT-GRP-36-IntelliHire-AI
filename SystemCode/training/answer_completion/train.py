"""Responsibilities: run Kaggle CPU training and honest held-out evaluation for semantic rough filtering.

Implementation: use a frozen pinned INT8 encoder and train a logistic head; select the threshold on
validation data only.
Related Modules: staged completion_gate.py supplies the exact serving encoder; experiment.json fixes
conditions, data.jsonl stores source-group splits, and challenge.jsonl contains separate examples.

Declaration Index:
- install_dependencies: install the experiment's exact training runtime versions in Kaggle.
- metrics: compute per-cohort candidate recall, rejection, and confusion matrices without relabeling.
- choose_threshold: choose the highest validation threshold meeting each cohort's recall target.
- resolve_source: locate the unique explicitly identified experiment in Kaggle's mounted input tree.
- train: check integrity, fit on training only, evaluate once, and export a checksum-bound bundle.
- main: resolve dataset/output arguments and run the fixed experiment without retries.

Variable Index:
None

Constraints:
No parameter search on test/challenge sets, fallback, automatic retry, or label correction. Failed
release criteria produce an unapproved artifact and nonzero exit. Synthetic evaluation cannot establish
real microphone/interview accuracy. The private Kaggle input contains no application keys."""

import argparse
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
import time
import unicodedata
import warnings
from pathlib import Path

import numpy as np


def install_dependencies(config):
    """Inputs: precommitted experiment versions. Outputs: installed Kaggle-only packages.
    Logic: invoke pip with an argv list; errors abort. Application environment is not modified.
    """
    subprocess.run([
        sys.executable, "-m", "pip", "install", "--quiet",
        "onnxruntime==" + config["onnxruntime_version"],
        "tokenizers==" + config["tokenizers_version"],
        "scikit-learn==" + config["sklearn_version"],
    ], check=True)


def metrics(records, scores, threshold):
    """Inputs: labeled evaluation records, candidate probabilities and validation-only threshold.
    Outputs: overall and per-language matrices/recall/rejection/precision/pass rate.
    Logic: 1 is allowed to Qwen; 0 is locally rejected. Labels are synthetic scenario labels.
    """
    result = {}
    for language in ("overall", "zh", "en"):
        indices = [i for i, item in enumerate(records) if (
            language == "overall" or item["language"] == language
        )]
        y = np.asarray([records[i]["label"] for i in indices])
        predicted = np.asarray([scores[i] >= threshold for i in indices])
        tp, fn = int(((y == 1) & predicted).sum()), int(((y == 1) & ~predicted).sum())
        fp, tn = int(((y == 0) & predicted).sum()), int(((y == 0) & ~predicted).sum())
        if not tp + fn or not tn + fp:
            raise ValueError("Each evaluation language must contain both classes")
        result[language] = {
            "n": len(indices), "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "recall": tp / (tp + fn), "negative_rejection": tn / (tn + fp),
            "precision": tp / (tp + fp) if tp + fp else 0,
            "qwen_pass_fraction": (tp + fp) / len(indices),
        }
    return result


def choose_threshold(records, scores, target):
    """Inputs: validation records/probabilities and fixed minimum recall for each language.
    Outputs: maximum eligible threshold. Logic: candidates are validation scores only.
    Constraints: neither held-out test nor challenge scores are inspected here.
    """
    for value in sorted(set(float(x) for x in scores), reverse=True):
        if value <= 0 or value >= 1:
            continue
        report = metrics(records, scores, value)
        if all(report[lang]["recall"] >= target for lang in ("zh", "en")):
            return value
    raise ValueError("No valid threshold meets the fixed validation recall target")


def train(source, output, config):
    """Inputs: staged private source directory, output directory and fixed experiment config.
    Outputs: encoder/tokenizer/head/manifest/report ZIP; nonzero exit for unmet release criteria.
    Logic: source groups and normalized exact duplicates are checked before encoding. Fit only
    training rows; choose threshold on validation, then evaluate test/challenge once. Record exact
    versions/hashes and latency; preserve failed results without changing conditions.
    """
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression

    sys.path.insert(0, str(source))
    from completion_gate import SemanticEncoder, file_sha256

    records = [json.loads(line) for line in (source / "data.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()]
    challenge = [json.loads(line) for line in (source / "challenge.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()]
    normalized = [unicodedata.normalize("NFKC", item["text"]).casefold() for item in records]
    if len(set(normalized)) != len(normalized):
        raise ValueError("Duplicate generated records would contaminate evaluation")
    if any(unicodedata.normalize("NFKC", item["text"]).casefold() in set(normalized)
           for item in challenge):
        raise ValueError("Challenge overlaps generated records")
    group_splits = {}
    for item in records:
        previous = group_splits.setdefault(item["group_id"], item["split"])
        if previous != item["split"] or item["label"] not in (0, 1):
            raise ValueError("Invalid labels or source-group leakage")
    encoder = SemanticEncoder(source)
    all_records = records + challenge
    batches = []
    for start in range(0, len(all_records), config["batch_size"]):
        batches.append(encoder.encode([
            item["text"] for item in all_records[start:start + config["batch_size"]]
        ]))
        print("encoded", min(start + config["batch_size"], len(all_records)), flush=True)
    features = np.concatenate(batches)
    indices = {split: [i for i, item in enumerate(records) if item["split"] == split]
               for split in ("train", "validation", "test")}
    model = LogisticRegression(
        C=config["logistic_C"], max_iter=config["logistic_max_iter"],
        class_weight="balanced", solver="lbfgs", random_state=config["seed"],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(features[indices["train"]], [records[i]["label"] for i in indices["train"]])
    if list(model.classes_) != [0, 1]:
        raise ValueError("Classifier must learn the exact binary intent label order")
    scores = model.predict_proba(features)[:, 1]
    validation = [records[i] for i in indices["validation"]]
    threshold = choose_threshold(
        validation, scores[indices["validation"]], config["validation_recall_target_per_language"]
    )
    evaluation = {
        "validation": metrics(validation, scores[indices["validation"]], threshold),
        "test": metrics([records[i] for i in indices["test"]], scores[indices["test"]], threshold),
        "challenge": metrics(challenge, scores[len(records):], threshold),
    }
    approved = (
        all(evaluation["test"][lang]["recall"] >= config["test_recall_min_per_language"]
            for lang in ("zh", "en"))
        and evaluation["test"]["overall"]["negative_rejection"]
        >= config["test_negative_rejection_min"]
        and all(evaluation["challenge"][lang]["recall"]
                >= config["challenge_recall_min_per_language"] for lang in ("zh", "en"))
    )
    latencies = []
    for item in challenge:
        started = time.perf_counter()
        encoder.encode([item["text"]])
        latencies.append((time.perf_counter() - started) * 1000)
    versions = {name: importlib.metadata.version(name) for name in
                ("numpy", "onnxruntime", "tokenizers", "scikit-learn")}
    report = {
        "config": config, "versions": versions, "platform": platform.platform(),
        "evaluation": evaluation, "release_approved": approved,
        "threshold": threshold, "threshold_source": "validation_only",
        "labels": "synthetic scenario labels; no independent human gold or real ASR evaluation",
        "cpu_encoding_latency_ms": {
            "p50": float(np.percentile(latencies, 50)),
            "p95": float(np.percentile(latencies, 95)), "n": len(latencies),
        },
        "held_out_predictions": [
            {"id": item["id"], "language": item["language"], "label": item["label"],
             "score": float(scores[i]), "passed": bool(scores[i] >= threshold)}
            for i, item in enumerate(all_records)
            if i >= len(records) or item["split"] == "test"
        ],
    }
    output.mkdir(parents=True, exist_ok=False)
    for name in ("encoder.onnx", "tokenizer.json", "experiment.json", "provenance.json"):
        shutil.copy2(source / name, output / name)
    (output / "head.json").write_text(json.dumps({
        "weights": model.coef_[0].tolist(), "intercept": float(model.intercept_[0]),
        "threshold": threshold,
    }, indent=2), encoding="utf-8")
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps({
        "format": config["format"], "experiment_id": config["experiment_id"],
        "base_model": config["base_model"], "base_revision": config["base_revision"],
        "release_approved": approved,
        "sha256": {name: file_sha256(output / name)
                   for name in ("encoder.onnx", "tokenizer.json", "head.json")},
        "data_sha256": file_sha256(source / "data.jsonl"), "evaluation": evaluation,
    }, indent=2), encoding="utf-8")
    shutil.make_archive(str(output), "zip", output)
    print(json.dumps({"threshold": threshold, "evaluation": evaluation,
                      "release_approved": approved}, indent=2), flush=True)
    if not approved:
        raise RuntimeError("Fixed release criteria failed; artifact preserved but not deployable")


def main():
    """Inputs: --input/--output paths, defaulting to this private Kaggle dataset/run directory.
    Outputs: trained artifact or explicit error. Logic: install pins and run exactly one experiment.
    Constraints: no key/environment loading and no automatic retraining on evaluation failures.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/artifact"))
    args = parser.parse_args()
    source = args.input if args.input is not None else resolve_source()
    config = json.loads((source / "experiment.json").read_text(encoding="utf-8"))
    install_dependencies(config)
    train(source, args.output, config)


def resolve_source():
    """Inputs: Kaggle's mounted input tree. Outputs: unique directory for this exact experiment ID.
    Logic: discover mount layout rather than guessing a legacy flat path; reject ambiguity/missing
    inputs. Constraints: no alternate dataset, model, split, or experiment is selected on failure.
    """
    matches = [p.parent for p in Path("/kaggle/input").rglob("experiment.json") if json.loads(
        p.read_text(encoding="utf-8")
    ).get("experiment_id") == "semantic-completion-v1-20261003"]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one mounted semantic experiment, found {len(matches)}")
    print("source_directory", matches[0], flush=True)
    return matches[0]


if __name__ == "__main__":
    main()
