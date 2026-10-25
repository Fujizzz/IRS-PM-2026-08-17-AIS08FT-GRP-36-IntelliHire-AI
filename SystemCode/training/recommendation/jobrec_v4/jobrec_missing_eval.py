"""职责：评估v4固定预测并保存查询级证据、区间和工程候选门槛。
实现：沿用v1主指标；显式记录并列与正例排名；复用v2加权AP进行配对查询bootstrap。
关联：jobrec_missing定义协议；experiment和verify调用analyze，不在此训练或选择模型。
目录：query_details、summarize_strata、analyze。
变量索引：LOGGER为阶段日志；CONTRASTS为预登记模型差值；实例状态无。
约束：同任务跨模型/场景共用重采样；全负查询只从排名均值排除；诊断不证明公平性。
"""

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
    """功能：记录每查询排名及正例位置；输入仅该任务已观察行/分数/任务/资料可用数。
    输出查询表和正例表；逻辑：稳定降序保持历史并列口径，k截断与历史一致。
    约束：记录并列率；全负查询指标为NaN保留在表中，排序位置为1起始且依赖同分行序。
    """
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
    """功能：对同一正例比较遮蔽前后排名；输入某场景与clean的正例表；输出按可用数分层摘要。
    逻辑：按user/item一一合并后汇总人数/正例数/排名变化/命中变化；约束：候选池不删减。
    """
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
    """功能：完整分析下载或远端固定预测；输入测试配对/场景模型预测/场景特征/输出目录。
    输出metrics字典并保存查询证据/区间/门槛；逻辑：逐场景主指标，r0共享2000次查询抽样。
    约束：无训练或选模；预测须有限；r1-r4仅敏感性分析；工程门槛不等于上线许可。
    """
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
