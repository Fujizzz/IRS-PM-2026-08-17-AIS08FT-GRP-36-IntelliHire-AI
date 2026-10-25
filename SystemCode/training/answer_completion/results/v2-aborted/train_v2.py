"""Responsibilities: run one approved Kaggle GPU encoder-finetuning experiment and CPU release audit.

Implementation: fine-tune pinned multilingual MiniLM and a linear binary head, select the epoch by
validation weighted BCE, export INT8 ONNX, calibrate on validation with single-tail serving inputs,
then evaluate the new test/challenge once. Report the V1 challenge separately as a known regression.
Related Modules: train supplies unchanged metric/threshold helpers; completion_gate supplies the exact
offline serving transform; experiment-v2.json fixes all experimental conditions in advance.

Declaration Index:
- install_dependencies: install exact experiment library pins without modifying the application environment.
- read_records: load UTF-8 JSONL without relabeling, omission, or text transformation.
- validate_data: reject duplicates, source leakage, invalid labels, and challenge overlap.
- tensor_batch: apply serving tokenizer/tail limits, then make CUDA tensors for training.
- fit_network: fine-tune five epochs; choose minimum validation loss without reading held-out text.
- fit_network.Network: own the contextual encoder and linear head with masked mean/L2 pooling.
- fit_network.Network.__init__: load the immutable pretrained revision and initialize the learned head.
- fit_network.Network.forward: return an own-answer-ending logit from attended contextual tokens.
- export_encoder: export a fixed batch-one encoder and quantize it with declared INT8 parameters.
- export_encoder.HiddenStates: expose selected encoder hidden states to ONNX.
- export_encoder.HiddenStates.__init__: register the selected encoder as owned export state.
- export_encoder.HiddenStates.forward: return hidden states from three token-tensor inputs.
- evaluate_export: use actual serving CPU features, calibrate validation, evaluate once, and package.
- run: check CUDA/data contracts and orchestrate the single experiment.
- main: find the unique V2 Kaggle mount or explicit input, install pins, and run once.

Variable Index:
None

Constraints: CUDA is required; there is no CPU fallback, retry, held-out threshold tuning, or acceptance
relaxation. An unapproved bundle is preserved and exits nonzero. Synthetic teacher labels are not
independent human gold; this experiment cannot establish microphone accuracy."""

import argparse
import importlib.metadata
import json
import platform
import random
import shutil
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import torch
from torch import nn


def install_dependencies(config):
    """Inputs: predeclared exact library versions. Outputs: Kaggle-only installed dependencies.
    Logic: argv-based pip invocation, no retries or alternate versions; failures propagate.
    Constraints: torch/CUDA come from the Kaggle image and their actual versions are recorded.
    """
    subprocess.run([
        sys.executable, "-m", "pip", "install", "--quiet",
        "transformers==" + config["transformers_version"],
        "tokenizers==" + config["tokenizers_version"],
        "onnx==" + config["onnx_version"],
        "onnxruntime==" + config["onnxruntime_version"],
    ], check=True)


def read_records(path):
    """Inputs: JSONL path. Outputs: ordered parsed records; malformed bytes/JSON fail explicitly.
    Logic: preserve all records and teacher labels, including difficult semantic counterexamples.
    """
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def validate_data(records, challenge, regression):
    """Inputs: split-bearing dataset and new/known contrast sets. Outputs: None or integrity error.
    Logic: Unicode exact deduplication, disjoint groups/splits and bilingual binary-label coverage.
    Constraints: neither held-out set may overlap training; no rows are repaired or discarded here.
    """
    normalized = [unicodedata.normalize("NFKC", x["text"]).casefold() for x in records]
    if len(set(normalized)) != len(normalized):
        raise ValueError("Duplicate training/evaluation records")
    groups = {}
    for item in records:
        if (groups.setdefault(item["group_id"], item["split"]) != item["split"]
                or item["label"] not in (0, 1) or item["language"] not in ("zh", "en")):
            raise ValueError("Invalid source-group split or label/language")
    for name, rows in (("new challenge", challenge), ("known regression", regression)):
        values = [unicodedata.normalize("NFKC", x["text"]).casefold() for x in rows]
        if len(set(values)) != len(values) or set(values) & set(normalized):
            raise ValueError(f"{name} contains duplicates or dataset overlap")
        for language in ("zh", "en"):
            if {x["label"] for x in rows if x["language"] == language} != {0, 1}:
                raise ValueError(f"{name} lacks binary coverage for {language}")
    if {x["text"] for x in challenge} & {x["text"] for x in regression}:
        raise ValueError("New challenge duplicates the known regression")


def tensor_batch(tokenizer, rows, device):
    """Inputs: shared low-level tokenizer, raw rows and explicit CUDA device. Outputs: tensors/y.
    Logic: same stripped last-400-character tail and left-128-token truncation as serving; batched
    padding is used only during floating-point training. Actual release audit always uses batch one.
    """
    tokens = tokenizer.encode_batch([x["text"].strip()[-400:] for x in rows])
    inputs = {name: torch.tensor(values, dtype=torch.long, device=device) for name, values in {
        "input_ids": [x.ids for x in tokens],
        "attention_mask": [x.attention_mask for x in tokens],
        "token_type_ids": [x.type_ids for x in tokens],
    }.items()}
    labels = torch.tensor([x["label"] for x in rows], dtype=torch.float32, device=device)
    return inputs, labels


def fit_network(source, work, config, training, validation):
    """Inputs: fixed source/config and train/validation rows only. Outputs: chosen network/history.
    Logic: AdamW with separate encoder/head learning rates, class-balanced BCE, AMP and declared
    clipping. Select the lowest validation weighted BCE among all declared epochs, with no patience
    or test-based selection. Save the selected state after each strict validation improvement.
    Constraints: pretrained code is not trusted remotely; download is bound to a full commit SHA.
    """
    from tokenizers import Tokenizer
    from transformers import AutoModel

    class Network(nn.Module):
        """Functionality: learn a semantic own-answer-ending score from the full contextual tail.
        Logic: attended mean pooling and unit normalization connect encoder updates to binary BCE.
        Constraints: all encoder parameters train; head has exactly 384 weights and one intercept.
        """

        def __init__(self):
            """Inputs: enclosing fixed base model/revision. Outputs: initialized encoder/head.
            Logic: eager attention makes ONNX export explicit; no alternate checkpoint is loaded.
            """
            super().__init__()
            self.encoder = AutoModel.from_pretrained(
                config["base_model"], revision=config["base_revision"],
                trust_remote_code=False, attn_implementation="eager",
            )
            self.head = nn.Linear(384, 1)

        def forward(self, **inputs):
            """Inputs: token tensors with attention mask. Outputs: one logit per candidate tail.
            Logic: perform pooling/normalization in float32 even under encoder AMP, excluding pads.
            Constraints: special attended tokens participate, matching source and serving contracts.
            """
            hidden = self.encoder(**inputs).last_hidden_state.float()
            mask = inputs["attention_mask"].float().unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1e-9)
            return self.head(torch.nn.functional.normalize(pooled, dim=1)).squeeze(-1).float()

    tokenizer = Tokenizer.from_file(str(source / "tokenizer.json"))
    tokenizer.enable_truncation(max_length=128, direction="left")
    pad_id = tokenizer.token_to_id("<pad>")
    if pad_id is None:
        raise ValueError("Tokenizer is missing its required padding token")
    tokenizer.enable_padding(pad_id=pad_id, pad_token="<pad>")
    device = torch.device("cuda:0")
    network = Network().to(device)
    optimizer = torch.optim.AdamW([
        {"params": network.encoder.parameters(), "lr": config["encoder_learning_rate"]},
        {"params": network.head.parameters(), "lr": config["head_learning_rate"]},
    ], weight_decay=config["weight_decay"])
    positive = sum(x["label"] for x in training)
    if not 0 < positive < len(training):
        raise ValueError("Training requires both classes")
    pos_weight = torch.tensor((len(training) - positive) / positive, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scaler = torch.amp.GradScaler("cuda")
    rng = random.Random(config["seed"])
    history, best_loss = [], float("inf")
    checkpoint = work / "selected.pt"
    for epoch in range(1, config["epochs"] + 1):
        started = time.perf_counter()
        network.train()
        order = list(training)
        rng.shuffle(order)
        train_loss = 0.0
        for start in range(0, len(order), config["batch_size"]):
            rows = order[start:start + config["batch_size"]]
            inputs, labels = tensor_batch(tokenizer, rows, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = criterion(network(**inputs), labels)
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite training loss at epoch={epoch} start={start}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(network.parameters(), config["gradient_clip_norm"],
                                     error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * len(rows)
            if start % (config["batch_size"] * 25) == 0:
                print("train_progress", epoch, start, "of", len(order), flush=True)
        network.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for start in range(0, len(validation), config["batch_size"]):
                rows = validation[start:start + config["batch_size"]]
                inputs, labels = tensor_batch(tokenizer, rows, device)
                validation_loss += criterion(network(**inputs), labels).item() * len(rows)
        validation_loss /= len(validation)
        if not np.isfinite(validation_loss):
            raise ValueError("Nonfinite validation loss")
        selected = validation_loss < best_loss
        if selected:
            best_loss = validation_loss
            torch.save(network.state_dict(), checkpoint)
        entry = {"epoch": epoch, "train_weighted_bce": train_loss / len(training),
                 "validation_weighted_bce": validation_loss, "selected": selected,
                 "seconds": time.perf_counter() - started}
        history.append(entry)
        (work / "epoch-history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print("epoch_result", json.dumps(entry), flush=True)
    network.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    return network.eval().cpu(), history


def export_encoder(network, source, output, work, config):
    """Inputs: selected network and explicit export/quantization config. Outputs: CPU INT8 files.
    Logic: fixed batch one, dynamic sequence length and three integer inputs; only declared weight
    operators are quantized. An invalid export/quantization aborts, with no alternate precision.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from tokenizers import Tokenizer

    class HiddenStates(nn.Module):
        """Functionality: expose the encoder's sole serving output without classifier serialization.
        Logic: return contextual hidden states; serving owns masked pooling and the JSON head.
        Constraints: fixed batch dimension one prevents quantization calibration/serving drift.
        """

        def __init__(self, encoder):
            """Inputs: selected offline encoder. Outputs: registered export state; no downloads.
            Logic: reuse already selected trained weights without initialization or mutation.
            """
            super().__init__()
            self.encoder = encoder

        def forward(self, input_ids, attention_mask, token_type_ids):
            """Inputs: three batch-one integer tensors. Outputs: [1, tokens, 384] hidden states.
            Logic: eager encoder forward without dropout; no language or keyword-based branches.
            """
            return self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                                token_type_ids=token_type_ids).last_hidden_state

    tokenizer = Tokenizer.from_file(str(source / "tokenizer.json"))
    token = tokenizer.encode("A fictional interview answer for tracing.")
    example = tuple(torch.tensor([value], dtype=torch.long) for value in
                    (token.ids, token.attention_mask, token.type_ids))
    float_path = work / "encoder-float.onnx"
    names = ["input_ids", "attention_mask", "token_type_ids"]
    torch.onnx.export(
        HiddenStates(network.encoder).eval(), example, str(float_path),
        input_names=names, output_names=["last_hidden_state"], opset_version=17,
        dynamic_axes={name: {1: "sequence"} for name in names + ["last_hidden_state"]},
        dynamo=False, external_data=False,
    )
    quantize_dynamic(
        str(float_path), str(output / "encoder.onnx"),
        weight_type=QuantType[config["quantization_weight_type"]],
        op_types_to_quantize=config["quantization_ops"],
        per_channel=config["quantization_per_channel"],
        reduce_range=config["quantization_reduce_range"],
    )
    shutil.copy2(source / "tokenizer.json", output / "tokenizer.json")
    print("export_complete", (output / "encoder.onnx").stat().st_size, flush=True)


def evaluate_export(network, source, output, config, records, challenge, regression, history):
    """Inputs: exported INT8 bundle, chosen head and fixed data/config/history. Outputs: audit.
    Logic: actual single-tail serving embeddings for all splits; validation-only threshold followed
    by one new test/challenge evaluation. Record known V1 regression separately from acceptance.
    Constraints: failures are preserved with approval=false; no test-based threshold changes.
    """
    sys.path.insert(0, str(source))
    from completion_gate import SemanticEncoder, file_sha256
    from train import choose_threshold, metrics

    encoder = SemanticEncoder(output)
    weights = network.head.weight.detach().numpy()[0].astype(np.float64)
    intercept = float(network.head.bias.detach().numpy()[0])
    all_rows = records + challenge + regression
    vectors = []
    for start in range(0, len(all_rows), 32):
        vectors.append(encoder.encode([x["text"] for x in all_rows[start:start + 32]]))
        print("audit_encoded", min(start + 32, len(all_rows)), flush=True)
    features = np.concatenate(vectors)
    logits = features @ weights + intercept
    scores = np.exp(-np.logaddexp(0, -logits))
    indices = {split: [i for i, x in enumerate(records) if x["split"] == split]
               for split in ("train", "validation", "test")}
    val = [records[i] for i in indices["validation"]]
    threshold = choose_threshold(val, scores[indices["validation"]],
                                 config["validation_recall_target_per_language"])
    challenge_end = len(records) + len(challenge)
    evaluation = {
        "validation": metrics(val, scores[indices["validation"]], threshold),
        "test": metrics([records[i] for i in indices["test"]], scores[indices["test"]], threshold),
        "challenge": metrics(challenge, scores[len(records):challenge_end], threshold),
        "known_v1_regression": metrics(regression, scores[challenge_end:], threshold),
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
    for row in challenge:
        started = time.perf_counter()
        encoder.encode([row["text"]])
        latencies.append((time.perf_counter() - started) * 1000)
    together = encoder.encode([x["text"] for x in challenge])
    individual = np.concatenate([encoder.encode([x["text"]]) for x in challenge])
    if not np.array_equal(together, individual):
        raise ValueError("Serving and multi-row audit embeddings differ despite batch-one contract")
    versions = {name: importlib.metadata.version(name) for name in
                ("numpy", "torch", "transformers", "tokenizers", "onnx", "onnxruntime")}
    report = {
        "config": config, "versions": versions, "platform": platform.platform(),
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        "epoch_history": history, "evaluation": evaluation, "release_approved": approved,
        "threshold": threshold, "threshold_source": "exported_INT8_single_tail_validation_only",
        "labels": "synthetic blind teacher labels; no independent human gold or real ASR",
        "batch_consistency": {"exactly_equal": True, "n": len(challenge)},
        "cpu_encoding_latency_ms": {
            "p50": float(np.percentile(latencies, 50)),
            "p95": float(np.percentile(latencies, 95)), "n": len(latencies),
        },
        "held_out_predictions": [
            {"id": x["id"], "language": x["language"], "label": x["label"],
             "score": float(scores[i]), "passed": bool(scores[i] >= threshold)}
            for i, x in enumerate(all_rows) if i >= len(records) or x["split"] == "test"
        ],
    }
    (output / "head.json").write_text(json.dumps({
        "weights": weights.tolist(), "intercept": intercept, "threshold": threshold,
    }, indent=2), encoding="utf-8")
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name in ("experiment.json", "provenance.json"):
        shutil.copy2(source / name, output / name)
    (output / "manifest.json").write_text(json.dumps({
        "format": config["format"], "experiment_id": config["experiment_id"],
        "base_model": config["base_model"], "base_revision": config["base_revision"],
        "release_approved": approved, "inference_batch_size": 1,
        "sha256": {name: file_sha256(output / name) for name in
                   ("encoder.onnx", "tokenizer.json", "head.json")},
        "data_sha256": file_sha256(source / "data.jsonl"), "evaluation": evaluation,
    }, indent=2), encoding="utf-8")
    shutil.make_archive(str(output), "zip", output)
    print("release_result", json.dumps({"threshold": threshold, "evaluation": evaluation,
                                        "release_approved": approved}), flush=True)
    if not approved:
        raise RuntimeError("Fixed V2 release criteria failed; unapproved bundle preserved")


def run(source, output, config):
    """Inputs: explicit V2 source/output/config. Outputs: finetuned approved or failed bundle.
    Logic: seed RNGs, require actual CUDA, validate dataset integrity, train then export/audit.
    Constraints: test/challenge text is never passed into fit_network; no parameter search occurs.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("Approved V2 experiment requires a CUDA accelerator")
    if (config["evaluation_inference_batch_size"] != 1 or config["mixed_precision"] != "float16"
            or config["checkpoint_selection"] != "minimum_validation_weighted_bce"):
        raise ValueError("V2 configuration violates its declared training/serving contract")
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    records = read_records(source / "data.jsonl")
    challenge = read_records(source / "challenge.jsonl")
    regression = read_records(source / "regression-v1.jsonl")
    validate_data(records, challenge, regression)
    output.mkdir(parents=True, exist_ok=False)
    work = output.parent / "training-state"
    work.mkdir(exist_ok=False)
    network, history = fit_network(source, work, config,
                                  [x for x in records if x["split"] == "train"],
                                  [x for x in records if x["split"] == "validation"])
    export_encoder(network, source, output, work, config)
    evaluate_export(network, source, output, config, records, challenge, regression, history)


def main():
    """Inputs: optional input/output CLI paths and Kaggle mount. Outputs: single experiment result.
    Logic: accept exactly one matching V2 dataset; pins install before transformer imports.
    Constraints: absence/ambiguity aborts rather than selecting another dataset or training mode.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/artifact"))
    args = parser.parse_args()
    matches = [args.input] if args.input is not None else [
        p.parent for p in Path("/kaggle/input").rglob("experiment.json") if json.loads(
            p.read_text(encoding="utf-8")
        ).get("experiment_id") == "semantic-completion-v2-20261003"
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one V2 source, found {len(matches)}")
    config = json.loads((matches[0] / "experiment.json").read_text(encoding="utf-8"))
    install_dependencies(config)
    print("experiment_start", config["experiment_id"], "source", matches[0], flush=True)
    run(matches[0], args.output, config)


if __name__ == "__main__":
    main()
