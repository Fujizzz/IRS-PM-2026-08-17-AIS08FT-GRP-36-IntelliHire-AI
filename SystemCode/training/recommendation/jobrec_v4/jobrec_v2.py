"""Responsibilities: measure feature and task-sharing contributions using fixed v1 data and embeddings.
Implementation: verify input hashes, compare four networks from the same initialization, and use paired
query bootstrap on test predictions.
Related Modules: jobrec_v1 is the original experiment.py embedded in the notebook; its source hash is fixed.

Declaration Index:
- verify_inputs: verify parent artifact identity.
- feature_view: remove the registered feature group for one comparison variant.
- IndependentHeads: define separate task networks initialized from the shared starting model.
- IndependentHeads.__init__: clone the shared trunk and each task's corresponding head.
- IndependentHeads.forward: return both independent task logits.
- masked_loss: compute loss only for observed labels.
- train_variant: run one controlled training comparison and score every pair.
- bootstrap_ap: compute paired AP under query-resampling counts.
- query_bootstrap: compute model intervals and preregistered paired differences.
- main: validate inputs, train four comparisons, evaluate, and save audit results.

Variable Index:
- CONFIG: inherited fixed v1 experiment configuration.
- PROTOCOL: new v2 ablation conditions.
- INPUT_HASHES: hashes binding the required parent inputs.
- LOGGER: stage-level logs without credentials or profile text.
- branches: instance state containing two networks with independent parameters.

Constraints: do not modify v1, split, seed, or hyperparameters; bootstrap intervals do not test
training-seed variability."""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

import jobrec_v1 as base

CONFIG = base.CONFIG
PROTOCOL = {
    "name": "jobrec-fixed-split-ablation-v2",
    "variants": ["shared_full", "shared_structured", "shared_vector", "independent_full"],
    "bootstrap_replicates": 2000,
    "bootstrap_seed": 42,
    "initialization": "reset seed42 immediately before every variant; clone identical shared trunk for independent branches",
    "feature_ablation": "zero removed blocks, retaining 2059 inputs and identical shared parameter counts",
    "checkpoint_selection": "same joint mean validation AP for all four variants",
    "comparison_scope": "v2-to-v2 controlled comparison; v1 RNG position before network initialization may differ",
}
INPUT_HASHES = {
    "config.json": "8dea69e2e61e7b5ce0812072f04eb17f931aad0c2fa97aeba2deb88e8db28513",
    "data_manifest.json": "9f692a0f8921625287408c9cbd5343823a2c612e50a43db0937667b6d0c73957",
    "pair_splits.csv": "37b134a5a2461481ce9d04318c6d05bf5025752173236b334db97cf130db63c0",
    "split_manifest.json": "dd228b8394d9d14e0bca0c77fadb07625d7f7ff3ee07472a2a93ed083cfd7eb1",
    "embeddings.npz": "c468440b592455f1503b022784c7ab44a7c4b9600ffe4a17e3f46f462ec2cc3b",
    "test_predictions.csv": "2045a94220c1d61bb19a3de4da26d10fe74eab5853d048e9eeb3bbb23b0b660b",
    "feature_scaler.json": "4ec782a0009581b88c7bcd21d824c410181350916cbda163b1680b31f53a5051",
}
LOGGER = logging.getLogger("jobrec_ablation")


def verify_inputs(folder: Path) -> None:
    """Functionality: verify the identity of parent experiment artifacts.
    Inputs: parent output directory. Outputs: none.
    Logic: compare SHA256 values for each required file and validate configuration.
    Constraints: reject updated parent artifacts even when filenames are unchanged."""
    for name, expected in INPUT_HASHES.items():
        actual = hashlib.sha256((folder / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Parent artifact changed: {name}: {actual}")
    if json.loads((folder / "config.json").read_text()) != CONFIG:
        raise ValueError("Parent config differs from embedded v1 source")
    LOGGER.info("Verified all %d parent artifact hashes", len(INPUT_HASHES))


def feature_view(features: np.ndarray, variant: str) -> np.ndarray:
    """Functionality: remove one feature group for a registered comparison.
    Inputs: N-by-2059 matrix and experiment name. Outputs: an independent copy.
    Logic: zero the first 2048 columns for the structured-only view or the final 11 columns for the
    embedding-only view.
    Constraints: preserve the source matrix. Layer widths and parameter counts remain the same, but
    zeroed inputs reduce the effective usable feature capacity."""
    if features.shape[1] != 2059 or variant not in PROTOCOL["variants"]:
        raise ValueError("Unexpected feature dimension or variant")
    result = features.copy()
    if variant == "shared_structured":
        result[:, :-11] = 0
    elif variant == "shared_vector":
        result[:, -11:] = 0
    return result


class IndependentHeads(torch.nn.Module):
    """Functionality: define independent task networks.
    Logic: each network copies the shared initial trunk and its corresponding task head.
    Constraints: preserve each task's layer width, approximately double total parameters, and interpret
    this as a sharing comparison rather than an equal-parameter comparison."""

    def __init__(self, template: base.DualHeadModel):
        """Functionality: create independent task branches.
        Inputs: initialized shared model. Outputs: a new IndependentHeads instance.
        Logic: deepcopy prevents parameter aliasing and does not consume new initialization random numbers.
        Constraints: only `branches` carries model state."""
        super().__init__()
        self.branches = torch.nn.ModuleList([
            torch.nn.Sequential(copy.deepcopy(template.trunk), copy.deepcopy(head))
            for head in template.heads
        ])

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Functionality: compute logits from the independent task branches.
        Inputs: B-by-2059 tensor. Outputs: B-by-2 logits.
        Logic: concatenate each branch's task output.
        Constraints: dropout masks are sampled independently in the two branches during training."""
        return torch.cat([branch(features) for branch in self.branches], dim=1)


def masked_loss(logits: torch.Tensor, labels: torch.Tensor, observed: torch.Tensor,
                positive_weights: torch.Tensor) -> torch.Tensor:
    """Functionality: calculate loss over observed labels only.
    Inputs: B-by-2 logits, labels, mask, and two positive-class weights. Outputs: scalar loss.
    Logic: average within each task's observed labels, then average across tasks, matching v1.
    Constraints: unknown positions must contain finite placeholder values; raise explicitly if neither task
    has an observed label."""
    element = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, labels, pos_weight=positive_weights, reduction="none")
    losses = [(element[:, c] * observed[:, c]).sum() / observed[:, c].sum()
              for c in range(2) if observed[:, c].any()]
    if not losses:
        raise ValueError("Batch has no observed task labels")
    return torch.stack(losses).mean()


def train_variant(pairs: pd.DataFrame, features: np.ndarray, variant: str, output: Path) -> np.ndarray:
    """Functionality: run one controlled network comparison.
    Inputs: pairs, feature matrix, variant name, and output directory. Outputs: N-by-2 scores.
    Logic: reset the same seed before each variant; retain minibatch order, loss, and joint validation
    selection; save the selected checkpoint.
    Constraints: select only within 12 epochs and never use test labels; v1's pretraining RNG position is
    not treated as part of this comparison."""
    random.seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])
    torch.cuda.manual_seed_all(CONFIG["seed"])
    template = base.DualHeadModel(features.shape[1])
    model = (IndependentHeads(template) if variant == "independent_full" else template).to("cuda:0")
    parameter_count = sum(p.numel() for p in model.parameters())
    initial = {k: value.cpu() for k, value in template.state_dict().items()}
    torch.save(initial, output / "initial_shared_template.pt")
    x = torch.from_numpy(feature_view(features, variant)).to("cuda:0")
    targets = pairs[["label_pref", "label_qual"]].to_numpy(dtype=np.float32)
    observed = np.isfinite(targets)
    y = torch.from_numpy(np.nan_to_num(targets, nan=0)).to("cuda:0")
    mask = torch.from_numpy(observed).to("cuda:0")
    train_ids = np.flatnonzero(pairs.split.eq("train"))
    val_ids = np.flatnonzero(pairs.split.eq("validation"))
    weights = []
    for column in range(2):
        labels = targets[train_ids, column]
        labels = labels[np.isfinite(labels)]
        weights.append(float((labels == 0).sum() / (labels == 1).sum()))
    positive_weights = torch.tensor(weights, device="cuda:0")
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    best_score, best_state, history = -1.0, None, []
    rng = np.random.default_rng(CONFIG["seed"])
    started = time.monotonic()
    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        losses, order = [], rng.permutation(train_ids)
        for start in range(0, len(order), CONFIG["training_batch_size"]):
            ids = order[start:start + CONFIG["training_batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            loss = masked_loss(model(x[ids]), y[ids], mask[ids], positive_weights)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        model.eval()
        with torch.inference_mode():
            val_scores = torch.sigmoid(model(x[val_ids])).cpu().numpy()
        aps = [float(average_precision_score(targets[val_ids, c][observed[val_ids, c]],
               val_scores[:, c][observed[val_ids, c]])) for c in range(2)]
        validation_mean = float(np.mean(aps))
        if not np.isfinite(validation_mean):
            raise ValueError("Non-finite validation AP")
        row = {"epoch": epoch, "mean_batch_loss": float(np.mean(losses)), "validation_ap": aps}
        history.append(row)
        LOGGER.info("%s epoch=%s", variant, row)
        if validation_mean > best_score:
            best_score, best_state, selected_epoch = validation_mean, copy.deepcopy(model.state_dict()), epoch
    model.load_state_dict(best_state)
    with torch.inference_mode():
        scores = torch.sigmoid(model(x)).cpu().numpy()
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite prediction")
    torch.save({"state_dict": {k: v.cpu() for k, v in best_state.items()}, "variant": variant,
                "config": CONFIG, "protocol": PROTOCOL}, output / "checkpoint.pt")
    base.write_json(output / "training_history.json", {"epochs": history, "selected_epoch": selected_epoch,
                    "validation_mean_ap": best_score, "parameter_count": parameter_count,
                    "positive_weights": weights, "training_seconds": time.monotonic() - started})
    del model, template, x, y, mask, optimizer, best_state
    gc.collect()
    torch.cuda.empty_cache()
    return scores


def bootstrap_ap(labels: np.ndarray, scores: np.ndarray, group_ids: np.ndarray,
                 counts: np.ndarray) -> np.ndarray:
    """Functionality: compute paired AP for query-resampled test sets.
    Inputs: labels, scores, group indices, and bootstrap query counts. Outputs: one AP per replicate.
    Logic: use query multiplicities as sample weights and compute precision steps across tied scores.
    Constraints: this is not mean query AP; reject replicates without positives; process 100 replicates per
    block to limit memory."""
    order = np.argsort(-scores, kind="stable")
    sorted_labels, sorted_scores = labels[order], scores[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(scores) - 1]
    values = []
    for start in range(0, len(counts), 100):
        weights = counts[start:start + 100, group_ids[order]].astype(float)
        positive = np.cumsum(weights * sorted_labels, axis=1)[:, ends]
        total = np.cumsum(weights, axis=1)[:, ends]
        if (positive[:, -1] == 0).any():
            raise ValueError("Bootstrap replicate has no positives")
        precision = np.divide(positive, total, out=np.zeros_like(positive), where=total > 0)
        increments = np.diff(np.column_stack([np.zeros(len(positive)), positive]), axis=1)
        values.extend(((increments * precision).sum(axis=1) / positive[:, -1]).tolist())
    return np.asarray(values)


def query_bootstrap(predictions: pd.DataFrame, model_names: list[str], output: Path) -> dict:
    """Functionality: quantify query-sampling uncertainty for fixed models.
    Inputs: test predictions, model names, and output directory. Outputs: interval reports.
    Logic: sample preference candidates and qualification jobs 2,000 times; reuse the same draws for all
    models within each task; report AP, nDCG@5, Recall@5, and preregistered paired differences.
    Constraints: percentile 95% intervals are exploratory and unadjusted for multiple comparisons.
    Preference results are conditional on the current test jobs and do not include training-seed or
    alternative-split variation. Exclude all-negative queries only from ranking means."""
    result = {"replicates": PROTOCOL["bootstrap_replicates"], "models": {}, "comparisons": {}}
    contrasts = [("shared_full", name) for name in (
        "shared_structured", "shared_vector", "independent_full", "structured_logistic")]
    for task in ("pref", "qual"):
        frame = predictions.loc[predictions["label_" + task].notna()].reset_index(drop=True)
        group_column = "user" if task == "pref" else "item"
        groups, ids = np.unique(frame[group_column], return_inverse=True)
        labels = frame["label_" + task].to_numpy()
        rng = np.random.default_rng(PROTOCOL["bootstrap_seed"])
        counts = rng.multinomial(len(groups), np.full(len(groups), 1 / len(groups)),
                                 size=PROTOCOL["bootstrap_replicates"])
        distributions = {}
        for name in model_names:
            scores = frame[name + "_" + task].to_numpy()
            ap = bootstrap_ap(labels, scores, ids, counts)
            query_scores = np.full((len(groups), 2), np.nan)
            for i in range(len(groups)):
                group = frame.loc[ids == i]
                if group["label_" + task].sum() > 0:
                    metrics = base.ranking_metrics(group, scores[ids == i], task)
                    query_scores[i] = [metrics["ndcg@5"], metrics["recall@5"]]
            valid = np.isfinite(query_scores[:, 0])
            denominator = counts[:, valid].sum(axis=1)
            if (denominator == 0).any():
                raise ValueError("Bootstrap replicate has no positive queries")
            rank = counts[:, valid] @ query_scores[valid] / denominator[:, None]
            distributions[name] = {"average_precision": ap, "ndcg@5": rank[:, 0], "recall@5": rank[:, 1]}
            result["models"].setdefault(name, {})[task] = {
                metric: np.quantile(values, [0.025, 0.975]).tolist()
                for metric, values in distributions[name].items()
            }
            LOGGER.info("Bootstrap completed task=%s model=%s queries=%d", task, name, len(groups))
        for left, right in contrasts:
            key = left + "_minus_" + right
            result["comparisons"].setdefault(key, {})[task] = {
                metric: {"ci95": np.quantile(distributions[left][metric] - distributions[right][metric],
                                             [0.025, 0.975]).tolist()}
                for metric in distributions[left]
            }
    base.write_json(output / "bootstrap.json", result)
    return result


def main() -> None:
    """Functionality: run the ablation and result audit.
    Inputs: mounted v1 artifacts and Kaggle GPU environment. Outputs: `ablation_v2` artifacts.
    Logic: verify parent identity, reconstruct pairs/features, train four variants, evaluate, and analyze
    uncertainty.
    Constraints: record failures and re-raise; do not change hyperparameters, retry, or fall back. Reuse
    verified embeddings rather than downloading or encoding them again."""
    output = Path("/kaggle/working/ablation_v2")
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(output / "execution.log")])
    stage, started = "parent_verification", time.monotonic()
    base.write_json(output / "protocol.json", {"base_config": CONFIG, "ablation": PROTOCOL, "input_hashes": INPUT_HASHES})
    try:
        candidates = list(Path("/kaggle/input").rglob("jobrec_v1/config.json"))
        if len(candidates) != 1:
            raise ValueError(f"Expected one mounted parent artifact, found {len(candidates)}")
        parent = candidates[0].parent
        verify_inputs(parent)
        stage = "environment_and_data"
        base.bootstrap(output)
        frames = base.download_data(output)
        if json.loads((output / "data_manifest.json").read_text()) != json.loads((parent / "data_manifest.json").read_text()):
            raise ValueError("Downloaded source data changed")
        pairs, _, _, _, structured = base.prepare_data(frames, output)
        pd.testing.assert_frame_equal(pairs, pd.read_csv(parent / "pair_splits.csv"), check_dtype=False)
        archived = np.load(parent / "embeddings.npz")
        features, _ = base.build_pair_features(pairs, (archived["candidates"], archived["jobs"]), structured, output)
        if json.loads((output / "feature_scaler.json").read_text()) != json.loads((parent / "feature_scaler.json").read_text()):
            raise ValueError("Feature preprocessing differs from v1")
        results, scores = {}, {}
        for variant in PROTOCOL["variants"]:
            stage = "training_" + variant
            destination = output / variant
            destination.mkdir()
            scores[variant] = train_variant(pairs, features, variant, destination)
        stage = "evaluation"
        test = pairs.loc[pairs.split.eq("test")].copy()
        for name, values in scores.items():
            results[name] = base.evaluate_scores(pairs, values, "test")
            test[name + "_pref"] = values[test.index, 0]
            test[name + "_qual"] = values[test.index, 1]
        previous = pd.read_csv(parent / "test_predictions.csv")
        pd.testing.assert_frame_equal(test[["user", "item"]].reset_index(drop=True), previous[["user", "item"]])
        for task in ("pref", "qual"):
            test["structured_logistic_" + task] = previous["structured_logistic_" + task].to_numpy()
        results["structured_logistic"] = base.evaluate_scores(test,
            test[["structured_logistic_pref", "structured_logistic_qual"]].to_numpy(), "test")
        test.to_csv(output / "test_predictions.csv", index=False)
        base.write_json(output / "metrics.json", results)
        stage = "query_bootstrap"
        query_bootstrap(test, list(results), output)
        base.write_json(output / "status.json", {"status": "complete", "total_seconds": time.monotonic() - started})
        LOGGER.info("ABLATION_COMPLETE %s", json.dumps(results))
    except Exception as error:
        base.write_json(output / "status.json", {"status": "failed", "stage": stage,
                        "error_type": type(error).__name__, "error": str(error)})
        LOGGER.exception("Ablation failed at stage=%s; no fallback", stage)
        raise


if __name__ == "__main__":
    main()
