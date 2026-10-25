"""职责：在固定 v1 数据与向量上验证特征和任务共享的贡献。
实现：校验输入哈希；同一起点比较四种网络；按查询配对 bootstrap 测试预测。
关联：jobrec_v1 是 notebook 内封装的原版 experiment.py，源码 SHA256 固定。
目录：verify_inputs、feature_view、IndependentHeads、IndependentHeads.__init__、
IndependentHeads.forward、masked_loss、train_variant、bootstrap_ap、query_bootstrap、main。
变量索引：CONFIG 为原实验配置；PROTOCOL 为新增消融配置；INPUT_HASHES 固定输入哈希；
LOGGER 为不含凭据/档案正文的阶段日志。实例 branches 存两个互不共享参数的网络。
约束：不修改 v1，不更换划分、训练种子或超参数；bootstrap 不代表训练随机性检验。
"""

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
    """功能：验证父实验产物身份；输入父目录；输出无，失败抛出异常。
    逻辑：逐文件比对 SHA256 并检查配置；约束：不接受同名但已更新的父产物。
    """
    for name, expected in INPUT_HASHES.items():
        actual = hashlib.sha256((folder / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Parent artifact changed: {name}: {actual}")
    if json.loads((folder / "config.json").read_text()) != CONFIG:
        raise ValueError("Parent config differs from embedded v1 source")
    LOGGER.info("Verified all %d parent artifact hashes", len(INPUT_HASHES))


def feature_view(features: np.ndarray, variant: str) -> np.ndarray:
    """功能：移除指定特征块；输入 N×2059矩阵和实验名；输出独立副本。
    逻辑：结构化组把前2048维归零，向量组把后11维归零；约束：原矩阵保持不变，
    保留相同层宽与参数量，但零输入意味着实际可用特征容量仍不同。
    """
    if features.shape[1] != 2059 or variant not in PROTOCOL["variants"]:
        raise ValueError("Unexpected feature dimension or variant")
    result = features.copy()
    if variant == "shared_structured":
        result[:, :-11] = 0
    elif variant == "shared_vector":
        result[:, -11:] = 0
    return result


class IndependentHeads(torch.nn.Module):
    """功能：将两任务拆为独立网络；逻辑：各自克隆同一初始 trunk 和对应 head。
    约束：每任务层宽不变，总参数约翻倍；这是共享与不共享对照，不是等总参数对照。
    """

    def __init__(self, template: base.DualHeadModel):
        """功能：创建独立分支；输入已初始化的共享模型；输出模块实例。
        逻辑：deepcopy 阻断参数别名且不消耗新的初始化随机数；约束：仅 branches 有状态。
        """
        super().__init__()
        self.branches = torch.nn.ModuleList([
            torch.nn.Sequential(copy.deepcopy(template.trunk), copy.deepcopy(head))
            for head in template.heads
        ])

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """功能：计算独立任务 logits；输入 B×2059；输出 B×2。
        逻辑：拼接两个分支输出；约束：训练时两分支各自抽取 dropout 掩码。
        """
        return torch.cat([branch(features) for branch in self.branches], dim=1)


def masked_loss(logits: torch.Tensor, labels: torch.Tensor, observed: torch.Tensor,
                positive_weights: torch.Tensor) -> torch.Tensor:
    """功能：仅使用观测标签计算损失；输入 B×2 logits/标签/mask 及2维正类权重。
    输出标量；逻辑：各有观测任务内部取均值，再对任务取均值，与 v1 相同。
    约束：未知位置标签必须已填有限占位值；两任务全无观测时明确报错。
    """
    element = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, labels, pos_weight=positive_weights, reduction="none")
    losses = [(element[:, c] * observed[:, c]).sum() / observed[:, c].sum()
              for c in range(2) if observed[:, c].any()]
    if not losses:
        raise ValueError("Batch has no observed task labels")
    return torch.stack(losses).mean()


def train_variant(pairs: pd.DataFrame, features: np.ndarray, variant: str, output: Path) -> np.ndarray:
    """功能：执行一次受控训练；输入配对/特征/组名/目录；输出所有配对的两任务分数。
    逻辑：每组训练前重设相同seed；同 minibatch 顺序、损失和联合验证选择；保存检查点。
    约束：仅在12轮内选择，不使用测试标签选模型；v1训练前的RNG位置不作为本组对照。
    """
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
    """功能：计算重复抽样查询后的配对AP；输入标签/分数/组索引/每次组抽样计数。
    输出每次抽样AP；逻辑：查询重复数作为样本权重，按相同分数整体计算precision阶梯。
    约束：不是query平均AP；拒绝无正例抽样，按100个replicate分块限制内存。
    """
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
    """功能：量化固定模型在测试查询抽样下的不确定性；输入测试预测/模型名/目录；输出区间。
    逻辑：偏好按候选人、资格按岗位抽样2000次，同一任务所有模型共用抽样；输出AP、
    nDCG@5和Recall@5及预先指定的配对差值区间。
    约束：百分位95%区间是探索性未校正多重比较；偏好结果条件于当前测试岗位集合，
    不覆盖训练种子或换岗位划分的方差；全负查询只从排名均值排除。
    """
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
    """功能：执行消融与结果审计；输入挂载的v1产物和Kaggle GPU环境；输出ablation_v2目录。
    逻辑：先验证身份及重新生成的配对/特征，再四组训练和统一评测，最后不确定性分析。
    约束：任何失败记录阶段并抛出；不改超参数、重试或回退。模型下载/编码直接复用已核验向量。
    """
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
