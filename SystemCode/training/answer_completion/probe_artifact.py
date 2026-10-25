"""Responsibilities: verify an approved bundle with the actual local serving implementation.

Implementation: load the checksum-bound gate, score fixed fictional holdouts, measure CPU latency,
and compare single-tail with multi-row requests. Optionally check Qwen routing on predeclared smoke text.
Related Modules: completion_gate is the real inference boundary; answer_completion owns Qwen.

Declaration Index:
- main: load the approved artifact, verify scoring consistency, and save local CPU and optional API probes.
- verify_qwen_cases: record real two-stage decisions/errors without retries or threshold changes.

Variable Index:
- SMOKE_CASES: fixed fictional routing checks; not independent gold or threshold-calibration data.

Constraints:
No threshold/model mutation, private .env edits, or production activation. This is an integration and
latency probe on this machine, not a new accuracy benchmark or real-microphone test."""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

SMOKE_CASES = (
    ("zh-end", "I use queue isolation for tasks. That is all I have to say about this question; we can move to the next one.", True),
    ("zh-task", "The project is complete. I will continue explaining the principles of queue isolation.", False),
    ("zh-quote", "The button displays 'Answer complete,' but I still need to explain why pressing it would lose data.", False),
    ("en-end", "I used a queue to isolate tasks. That is all I have to say for this question.",
     True),
    ("en-task", "The project is complete. I will explain how the queue isolates tasks next.",
     False),
    ("en-negate", "I am not finished answering yet; I still need to explain the retry policy.",
     False),
)


async def verify_qwen_cases(classifier, gate, cases, report, destination):
    """Inputs: real classifier/gate, fixed ID/text/label cases, mutable report and checkpoint path.
    Outputs: per-case decisions, errors, latencies and final success status saved incrementally.
    Logic: run each case once; a provider failure is recorded as an error, never a negative intent
    or replacement result. Continue independent cases for diagnosis, then fail the probe explicitly.
    Constraints: no retries, deadline changes, model/threshold tuning or microphone claims.
    """
    results = report["qwen_cases"] = []
    for key, text, expected in cases:
        score = gate.score(text)
        started = time.perf_counter()
        entry = {"id": key, "expected": expected, "gate_score": score,
                 "sent_to_qwen": score >= gate.threshold}
        try:
            entry["actual"] = await classifier.classify(text)
        except Exception as error:
            entry["error"] = type(error).__name__
            print("qwen_probe_error", key, type(error).__name__, flush=True)
        entry["elapsed_ms"] = (time.perf_counter() - started) * 1000
        results.append(entry)
        destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("qwen_probe_completed", key, "error" in entry, flush=True)
    report["qwen_error_count"] = sum("error" in x for x in results)
    report["qwen_mismatch_count"] = sum(x.get("actual") != x["expected"] for x in results
                                        if "error" not in x)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if report["qwen_error_count"]:
        raise RuntimeError("Qwen probe has explicit provider errors; all results preserved")


async def main():
    """Inputs: approved artifact/output paths, optional Qwen flags and explicit extra cases JSONL.
    Outputs: JSON CPU/contract report; configuration, network and numerical errors propagate.
    Logic: real gate loading, exact batch-one equality and comparison with trained predictions.
    Optional Qwen calls use only fixed fictional text, inherit existing timeouts and never retry.
    Constraints: smoke labels and runtime checks do not alter the declared release criteria.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qwen", action="store_true")
    parser.add_argument("--challenge-qwen", action="store_true")
    parser.add_argument("--cases", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    from agents.completion_gate import SemanticGate, file_sha256

    started = time.perf_counter()
    gate = SemanticGate(args.artifact)
    load_ms = (time.perf_counter() - started) * 1000
    challenge = [json.loads(line) for line in (Path(__file__).with_name("challenge-v2.jsonl"))
                 .read_text(encoding="utf-8").splitlines()]
    latencies, scores = [], []
    for item in challenge:
        started = time.perf_counter()
        scores.append(gate.score(item["text"]))
        latencies.append((time.perf_counter() - started) * 1000)
    batched = gate.encoder.encode([x["text"] for x in challenge])
    single = np.concatenate([gate.encoder.encode([x["text"]]) for x in challenge])
    if not np.array_equal(batched, single):
        raise ValueError("Single-tail and multi-row serving features differ")
    formal = json.loads((args.artifact / "report.json").read_text(encoding="utf-8"))
    predictions = {x["id"]: x for x in formal["held_out_predictions"]}
    deviations = [abs(scores[i] - predictions[x["id"]]["score"])
                  for i, x in enumerate(challenge)]
    flips = [x["id"] for i, x in enumerate(challenge)
             if bool(scores[i] >= gate.threshold) != predictions[x["id"]]["passed"]]
    if flips:
        raise ValueError(f"Local CPU gate decisions differ from formal audit for {flips}")
    report = {
        "artifact_manifest_sha256": file_sha256(args.artifact / "manifest.json"),
        "artifact_load_ms": load_ms, "threshold": gate.threshold,
        "single_multi_row_equal": True, "formal_decision_flips": flips,
        "formal_score_max_difference": max(deviations),
        "cpu_gate_ms": {"p50": float(np.percentile(latencies, 50)),
                        "p95": float(np.percentile(latencies, 95)), "n": len(challenge)},
        "scope": "This machine, synthetic fixed texts; no microphone or production activation",
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.qwen or args.challenge_qwen or args.cases:
        import os

        from agents.answer_completion import FlashCompletionClassifier

        load_dotenv(root / ".env")
        os.environ["ANSWER_COMPLETION_GATE_PATH"] = str(args.artifact.resolve())
        classifier = FlashCompletionClassifier()
        report["qwen_model"] = classifier.model
        report["qwen_timeout_seconds"] = classifier.timeout_seconds
        try:
            rows = [json.loads(line) for line in args.cases.read_text(
                encoding="utf-8"
            ).splitlines()] if args.cases else challenge
            cases = [(x["id"], x["text"], bool(x["label"])) for x in rows] if (
                args.challenge_qwen or args.cases
            ) else SMOKE_CASES
            await verify_qwen_cases(classifier, gate, cases, report, args.output)
        finally:
            await classifier.close()
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
