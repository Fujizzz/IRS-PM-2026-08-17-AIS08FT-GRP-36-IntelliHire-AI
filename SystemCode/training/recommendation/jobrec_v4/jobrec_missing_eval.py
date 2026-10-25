"""Responsibilities: evaluate fixed v4 predictions and save query-level evidence, intervals, and
engineering candidate-gate results.
Implementation: reuse v1 primary metrics, record ties and positive ranks explicitly, and reuse v2's
weighted AP for paired query bootstrap intervals.
Related Modules: jobrec_missing defines the protocol; experiment calls this evaluator after training.

Declaration Index:
- query_details: compute stable query rankings and positive-item positions.
- summarize_strata: summarize positive-pair changes by available-feature count.
- analyze: evaluate all fixed model/scenario predictions and write reports.

Variable Index:
- LOGGER: stage-level evaluation logging.
- CONTRASTS: preregistered model-score differences.
- Instance state: None.

Constraints: share bootstrap draws across models/scenarios within each task; exclude all-negative
queries only from ranking means; diagnostics do not establish fairness."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

import jobrec_v1 as base
import jobrec_v2 as previous
import jobrec_missing as spec

LOGGER = logging.getLogger("jobrec_v4_evaluation")
CONTRASTS = [("B", "A"), ("C", "B"), ("D", "B")] + [(m, "H") for m in "ABCD"]


def query_details(frame: pd.DataFrame, scores: np.ndarray, task: str,
                  available: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Functionality: record per-query rankings and positive-item positions.
    Inputs: observed rows for one task, their scores, task name, and available-feature counts.
    Outputs: query summary table and positive-pair table.
    Logic: use stable descending sort to preserve the historical tie rule and apply the same top-k limits.
    Constraints: report tie rates; retain all-negative queries with NaN ranking metrics; ranks start at one
    and depend on input order for tied scores."""
    frame = frame.reset_index(drop=True)
    rows, positives = [], []
    for query, indices in frame.groupby("user" if task == "pref" else "item", sort=True).indices.items():
        indices = np.asarray(indices)
        labels = frame.iloc[indices]["label_" + task].to_numpy()
        values = scores[indices]
        order = np.argsort(-values, kind="stable")
        ranked = labels[order]
        count = int(labels.sum())
        row = {"query": query, "pool": len(indices), "positives": count,
               "tied_fraction": float(1 - len(np.unique(values)) / len(values))}
        for k in (5, 10):
            limit = min(k, len(order))
            ideal = (1 / np.log2(np.arange(min(count, limit)) + 2)).sum()
            row[f"ndcg@{k}"] = float((ranked[:limit] / np.log2(np.arange(limit) + 2)).sum() / ideal) if count else np.nan
            row[f"recall@{k}"] = float(ranked[:limit].sum() / count) if count else np.nan
        rows.append(row)
        for rank, local in enumerate(order, 1):
            if labels[local] == 1:
                index = indices[local]
                positives.append({"user": frame.iloc[index].user, "item": frame.iloc[index].item,
                                  "query": query, "rank": rank, "hit5": int(rank <= 5),
                                  "available": int(available[index])})
    return pd.DataFrame(rows), pd.DataFrame(positives)


def summarize_strata(positive: pd.DataFrame, clean: pd.DataFrame) -> list[dict]:
    """Functionality: compare masked-profile ranks with the clean reference for the same positive pairs.
    Inputs: one scenario's positive table and the clean positive table.
    Outputs: summaries grouped by available-feature count.
    Logic: perform a one-to-one user/item join, then summarize users, positive pairs, rank changes, and
    hit changes.
    Constraints: do not remove candidate pairs; incomplete or duplicate joins raise an error."""
    joined = positive.merge(clean[["user", "item", "rank", "hit5"]], on=["user", "item"],
                            suffixes=("", "_clean"), validate="one_to_one")
    output = []
    for count, part in joined.groupby("available"):
        output.append({"available": int(count), "users": int(part.user.nunique()), "positive_pairs": len(part),
                       "mean_rank": float(part["rank"].mean()),
                       "mean_rank_worsening": float((part["rank"] - part.rank_clean).mean()),
                       "hit5": float(part.hit5.mean()), "hit5_change": float((part.hit5 - part.hit5_clean).mean())})
    return output


def analyze(pairs: pd.DataFrame, predictions: dict, feature_arrays: dict, output: Path) -> dict:
    """Functionality: analyze fixed predictions downloaded from Kaggle or produced by the experiment.
    Inputs: test pairs, model/scenario predictions, scenario feature arrays, and output directory.
    Outputs: metrics dictionary plus query evidence, bootstrap intervals, and candidate-gate reports.
    Logic: compute primary metrics per scenario and use shared 2,000-draw bootstrap samples for repeat 0.
    Constraints: no training or model selection occurs here; predictions must be finite; repeats 1–4 are
    sensitivity analysis only; an engineering gate is not release approval."""
    output.mkdir(parents=True, exist_ok=True)
    metrics, queries, positive_tables, all_query_rows, all_positive_rows = {}, {}, {}, [], []
    for scene, _ in feature_arrays.items():
        available = np.isfinite(feature_arrays[scene]).sum(axis=1)
        metrics[scene] = {}
        for model in spec.PROTOCOL["models"]:
            scores = predictions[scene + "__" + model]
            if scores.shape != (len(pairs), 2) or not np.isfinite(scores).all():
                raise ValueError(f"Invalid predictions: {scene}/{model}")
            metrics[scene][model] = base.evaluate_scores(pairs, scores, "test")
            for column, task in enumerate(("pref", "qual")):
                observed = pairs["label_" + task].notna().to_numpy()
                q, p = query_details(pairs.loc[observed], scores[observed, column], task, available[observed])
                key = scene + "__" + model + "__" + task
                queries[key], positive_tables[key] = q, p
                all_query_rows.append(q.assign(scene=scene, model=model, task=task))
                all_positive_rows.append(p.assign(scene=scene, model=model, task=task))
                for metric in ("ndcg@5", "ndcg@10", "recall@5", "recall@10"):
                    np.testing.assert_allclose(q[metric].mean(), metrics[scene][model][task][metric], rtol=0, atol=1e-12)
                metrics[scene][model][task]["mean_tied_fraction"] = float(q.tied_fraction.mean())
                metrics[scene][model][task]["queries_with_ties"] = int(q.tied_fraction.gt(0).sum())
        LOGGER.info("METRICS %s completed", scene)
    base.write_json(output / "metrics.json", metrics)
    pd.concat(all_query_rows, ignore_index=True).to_csv(output / "query_metrics.csv.gz", index=False)
    pd.concat(all_positive_rows, ignore_index=True).to_csv(output / "positive_ranks.csv.gz", index=False)
    strata = {}
    for scene in ("random_30_r0", "mixed_optional_r0", "optional_bundle_r0", "skills_only_r0"):
        for model in spec.PROTOCOL["models"]:
            for task in ("pref", "qual"):
                key = scene + "__" + model + "__" + task
                strata[key] = summarize_strata(positive_tables[key], positive_tables["clean_r0__" + model + "__" + task])
    base.write_json(output / "completeness_strata.json", strata)
    intervals, distributions, deltas = {}, {}, {}
    r0_scenes = [scene for scene in feature_arrays if scene.endswith("_r0")]
    for col, task in enumerate(("pref", "qual")):
        observed = pairs["label_" + task].notna().to_numpy()
        frame = pairs.loc[observed]
        groups, ids = np.unique(frame["user" if task == "pref" else "item"], return_inverse=True)
        counts = np.random.default_rng(42).multinomial(len(groups), np.full(len(groups), 1 / len(groups)), size=2000)
        for scene in r0_scenes:
            for model in spec.PROTOCOL["models"]:
                prefix = scene + "__" + model + "__" + task
                query = queries[prefix].set_index("query").loc[groups]
                valid = query.positives.gt(0).to_numpy()
                weights = counts[:, valid]
                denominators = weights.sum(axis=1)
                if (denominators == 0).any():
                    raise ValueError("No positive ranking queries in bootstrap draw")
                for metric in ("ndcg@5", "recall@5", "ndcg@10", "recall@10"):
                    distributions[prefix + "__" + metric] = weights @ query.loc[valid, metric].to_numpy() / denominators
                distributions[prefix + "__average_precision"] = previous.bootstrap_ap(
                    frame["label_" + task].to_numpy(), predictions[scene + "__" + model][observed, col], ids, counts)
            for metric in ("average_precision", "ndcg@5", "recall@5", "ndcg@10", "recall@10"):
                for left, right in CONTRASTS:
                    key = scene + "__" + left + "-" + right + "__" + task + "__" + metric
                    diff = distributions[f"{scene}__{left}__{task}__{metric}"] - distributions[f"{scene}__{right}__{task}__{metric}"]
                    deltas[key] = np.quantile(diff, [.025, .975]).tolist()
            LOGGER.info("BOOTSTRAP task=%s scene=%s complete", task, scene)
        for model in spec.PROTOCOL["models"]:
            key = "primary_mean__" + model + "__" + task + "__ndcg@5"
            distributions[key] = np.mean([distributions[f"{scene}_r0__{model}__{task}__ndcg@5"]
                                           for scene in spec.PROTOCOL["primary_scenarios"]], axis=0)
        for left, right in CONTRASTS:
            key = "primary_mean__" + left + "-" + right + "__" + task + "__ndcg@5"
            diff = distributions[f"primary_mean__{left}__{task}__ndcg@5"] - distributions[f"primary_mean__{right}__{task}__ndcg@5"]
            deltas[key] = np.quantile(diff, [.025, .975]).tolist()
    for key, values in distributions.items():
        intervals[key] = np.quantile(values, [.025, .975]).tolist()
    base.write_json(output / "bootstrap.json", {"seed": 42, "replicates": 2000, "intervals": intervals, "contrasts": deltas})
    np.savez_compressed(output / "bootstrap_draws.npz", **distributions)
    gates, sensitivity, clean_changes = {}, {}, {}
    for model in spec.PROTOCOL["models"]:
        gates[model] = {}
        for task in ("pref", "qual"):
            primary = float(np.mean([metrics[s + "_r0"][model][task]["ndcg@5"] for s in spec.PROTOCOL["primary_scenarios"]]))
            reference = float(np.mean([metrics[s + "_r0"]["H"][task]["ndcg@5"] for s in spec.PROTOCOL["primary_scenarios"]]))
            clean_ap = metrics["clean_r0"][model][task]["average_precision"] - metrics["clean_r0"]["H"][task]["average_precision"]
            clean_ndcg = metrics["clean_r0"][model][task]["ndcg@5"] - metrics["clean_r0"]["H"][task]["ndcg@5"]
            gates[model][task] = {"clean_ap_change": clean_ap, "clean_ndcg5_change": clean_ndcg,
                                  "primary_ndcg5": primary, "primary_gain": primary - reference,
                                  "point_estimate_gate_pass": bool(clean_ap >= -.02 and clean_ndcg >= -.02 and primary - reference >= .03)}
            for scene in r0_scenes:
                clean_changes[f"{scene}__{model}__{task}"] = {metric: metrics[scene][model][task][metric] - metrics["clean_r0"][model][task][metric]
                    for metric in ("average_precision", "ndcg@5", "recall@5", "recall@10")}
            for scenario in ("random_10", "random_30", "random_50", "mixed_optional", "job_random_30"):
                sensitivity[f"{scenario}__{model}__{task}"] = {metric: [min(metrics[f"{scenario}_r{r}"][model][task][metric] for r in range(5)),
                    max(metrics[f"{scenario}_r{r}"][model][task][metric] for r in range(5))]
                    for metric in ("average_precision", "ndcg@5", "recall@5")}
    base.write_json(output / "gates.json", gates)
    base.write_json(output / "mask_sensitivity.json", sensitivity)
    base.write_json(output / "change_from_clean.json", clean_changes)
    return metrics
