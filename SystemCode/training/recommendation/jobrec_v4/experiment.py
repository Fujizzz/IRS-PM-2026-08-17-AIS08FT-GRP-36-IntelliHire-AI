"""职责：执行已批准的v4缺失资料训练与历史基线核验。
实现：固定父产物哈希，重算资料遮蔽特征，训练8个树模型，保存预测并统一分析。
关联：jobrec_v1/v2复用历史协议；jobrec_missing定义缺失语义；jobrec_missing_eval分析输出。
目录：verify_history、prepare_scenes、train_model、main。
变量索引：LOGGER为无凭据阶段日志；其余实验参数由spec.PROTOCOL/RANKER_CONFIG提供。
约束：不改变历史模型、划分、标签、树超参数；失败保存阶段且抛出，无重试或回退。
"""

import hashlib
import importlib.metadata
import json
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

import jobrec_v1 as base
import jobrec_v2 as previous
import jobrec_missing as spec
import jobrec_missing_eval as evaluation

LOGGER = logging.getLogger("jobrec_v4")


def verify_history(folder: Path) -> None:
    """功能：核验v3历史树与结果身份；输入父目录；输出无。
    逻辑：逐文件SHA256比对冻结表；约束：任何不符即失败，不接受同名更新产物。
    """
    for name, expected in spec.HISTORICAL.items():
        if hashlib.sha256((folder / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"v3 historical artifact changed: {name}")
    LOGGER.info("Historical v3 artifacts verified")


def prepare_scenes(pairs: pd.DataFrame, users: pd.DataFrame, jobs: pd.DataFrame,
                   clean: np.ndarray, output: Path) -> dict:
    """功能：冻结所有缺失场景；输入完整数据与参考特征/输出目录；输出场景到全配对特征映射。
    逻辑：实体掩码→原始字段隐藏→特征重算，并保存实体ID、掩码和摘要；约束：clean须逐项相同。
    训练场景命名空间独立；原始档案不写输出；保存的特征含NaN，不用NaN填补标签。
    """
    data, masks, audits = {}, {}, {}
    settings = spec.scenarios() + [("training_mix", 0)]
    for scenario, repeat in settings:
        key = f"{scenario}_r{repeat}"
        um, jm, assignments = spec.make_masks(users, jobs, scenario, repeat,
                                              "training" if scenario == "training_mix" else "evaluation")
        hidden_u = spec.hide_profiles(users, um, spec.USER_BLOCKS)
        hidden_j = spec.hide_profiles(jobs, jm, spec.JOB_BLOCKS)
        matrix = spec.features(pairs, hidden_u, hidden_j)
        if scenario == "clean":
            np.testing.assert_array_equal(matrix, clean)
        data[key] = matrix
        masks[key + "__users"], masks[key + "__jobs"] = um, jm
        audits[key] = {"user_blocks_hidden": um.sum(axis=0).tolist(), "job_blocks_hidden": jm.sum(axis=0).tolist(),
                       "feature_missing_counts": np.isnan(matrix).sum(axis=0).tolist(),
                       "user_assignments": pd.Series(assignments).value_counts().to_dict()}
        if scenario == "training_mix":
            train_users = np.unique(pairs.loc[pairs.split.eq("train"), "u_idx"])
            audits[key]["labeled_training_user_assignments"] = pd.Series(np.asarray(assignments)[train_users]).value_counts().to_dict()
        LOGGER.info("FEATURES scene=%s missing=%s", key, audits[key]["feature_missing_counts"])
    np.savez_compressed(output / "masks.npz", **masks)
    users[["student_id"]].to_csv(output / "candidate_ids.csv", index=False)
    jobs[["job_id"]].to_csv(output / "job_ids.csv", index=False)
    base.write_json(output / "mask_audit.json", audits)
    np.savez_compressed(output / "training_features.npz", clean=data["clean_r0"], mixed=data["training_mix_r0"])
    return data


def train_model(model_name: str, pairs: pd.DataFrame, data: dict, folder: Path) -> list:
    """功能：训练一个预登记双向对照；输入模型名/配对/场景/模型目录；输出两个已裁剪Booster。
    逻辑：A用完整、BCD用同一混合资料；按任务查询成组；三种验证AP等权选20轮网格。
    约束：只用train拟合、validation选轮次；不复制配对；并列选更早；模型保存后重新加载。
    """
    import lightgbm as lgb

    outputs, history = [], {}
    train_features = spec.feature_view(data["clean_r0" if model_name == "A" else "training_mix_r0"], model_name)
    for task in ("pref", "qual"):
        label, group = "label_" + task, "user" if task == "pref" else "item"
        train = pairs.loc[pairs.split.eq("train") & pairs[label].notna()].sort_values(group, kind="stable")
        validation = pairs.loc[pairs.split.eq("validation") & pairs[label].notna()]
        sizes = train.groupby(group, sort=False).size().to_numpy()
        if sizes.sum() != len(train):
            raise ValueError("Invalid query grouping")
        model = lgb.LGBMRanker(**spec.RANKER_CONFIG)
        model.fit(train_features[train.index], train[label].astype(int), group=sizes)
        records = []
        for iteration in spec.PROTOCOL["tree_grid"]:
            aps = {}
            for scenario in spec.PROTOCOL["validation_scenarios"]:
                matrix = spec.feature_view(data[scenario + "_r0"][validation.index], model_name)
                scores = model.booster_.predict(matrix, num_iteration=iteration, num_threads=2)
                aps[scenario] = float(average_precision_score(validation[label], scores))
            records.append({"iteration": iteration, "scenario_ap": aps, "mean_ap": float(np.mean(list(aps.values())))})
        selected = max(records, key=lambda row: row["mean_ap"])
        target = folder / (task + ".txt")
        model.booster_.save_model(str(target), num_iteration=selected["iteration"])
        booster = lgb.Booster(model_file=str(target))
        outputs.append(booster)
        history[task] = {"selection": selected, "history": records, "training_pairs": len(train),
                         "training_queries": len(sizes), "actual_params": model.get_params(),
                         "features": train_features.shape[1]}
        LOGGER.info("TRAIN model=%s task=%s selected=%s", model_name, task, selected)
    base.write_json(folder / "history.json", history)
    return outputs


def main() -> None:
    """功能：运行完整v4；输入Kaggle父kernel、网络和已批准协议；输出jobrec_v4产物。
    逻辑：核验→环境→数据→clean复现→缺失诊断→8次拟合→评测；每阶段日志及失败状态落盘。
    约束：沿用历史环境引导；树模型仅CPU两线程，不声称使用GPU训练；不执行文本或面试模型。
    """
    output = Path("/kaggle/working/jobrec_v4")
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, force=True, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(output / "execution.log")])
    started, stage = time.monotonic(), "protocol"
    base.write_json(output / "protocol.json", spec.PROTOCOL)
    base.write_json(output / "protocol_hash.json", {"sha256": hashlib.sha256((output / "protocol.json").read_bytes()).hexdigest()})
    try:
        roots = list(Path("/kaggle/input").rglob("jobrec_v1/config.json"))
        histories = list(Path("/kaggle/input").rglob("jobrec_v3/lambdarank/pref.txt"))
        if len(roots) != 1 or len(histories) != 1:
            raise ValueError(f"Expected one v1/v3 parent; found {len(roots)}/{len(histories)}")
        parent, history_parent = roots[0].parent, histories[0].parent.parent
        previous.verify_inputs(parent)
        verify_history(history_parent)
        stage = "environment"
        base.bootstrap(output)
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "lightgbm==4.6.0"], check=True)
        import lightgbm as lgb

        base.write_json(output / "ranker_environment.json", {"lightgbm": importlib.metadata.version("lightgbm"),
                        "compute": "CPU, n_jobs=2; no embedding or neural training"})
        stage = "data"
        frames = base.download_data(output)
        if json.loads((output / "data_manifest.json").read_text()) != json.loads((parent / "data_manifest.json").read_text()):
            raise ValueError("Source data changed")
        pairs, users, jobs, _, clean = base.prepare_data(frames, output)
        pd.testing.assert_frame_equal(pairs, pd.read_csv(parent / "pair_splits.csv"), check_dtype=False)
        np.testing.assert_array_equal(spec.features(pairs, users, jobs), clean)
        test = pairs.loc[pairs.split.eq("test")].copy()
        test_ids = test.index.to_numpy()
        test.to_csv(output / "test_pairs.csv", index=False)
        models, predictions = {}, {}
        historical_folder = output / "H"
        historical_folder.mkdir()
        models["H"] = []
        for task in ("pref", "qual"):
            shutil.copy2(history_parent / "lambdarank" / (task + ".txt"), historical_folder / (task + ".txt"))
            models["H"].append(lgb.Booster(model_file=str(historical_folder / (task + ".txt"))))
        clean_scores = np.column_stack([model.predict(clean[test_ids], num_threads=2) for model in models["H"]])
        old = pd.read_csv(history_parent / "test_predictions.csv")
        pd.testing.assert_frame_equal(test[["user", "item"]].reset_index(drop=True), old[["user", "item"]])
        np.testing.assert_allclose(clean_scores, old[["lambdarank_taskwise_pref", "lambdarank_taskwise_qual"]].to_numpy(), rtol=1e-12, atol=1e-12)
        reference_metrics = json.loads((history_parent / "metrics.json").read_text())["lambdarank_taskwise"]
        actual = base.evaluate_scores(test, clean_scores, "test")
        for task in actual:
            for metric, value in actual[task].items():
                np.testing.assert_allclose(value, reference_metrics[task][metric], rtol=0, atol=1e-12)
        base.write_json(output / "baseline_verification.json", {"features_exact": True, "predictions_tolerance": 1e-12,
                        "metrics_tolerance": 1e-12, "metrics": actual, "history_hashes": spec.HISTORICAL})
        LOGGER.info("BASELINE clean features, predictions and metrics verified")
        stage = "mask_features"
        data = prepare_scenes(pairs, users, jobs, clean, output)
        scenes = {key: matrix[test_ids] for key, matrix in data.items() if key != "training_mix_r0"}
        np.savez_compressed(output / "test_features.npz", **scenes)
        stage = "historical_diagnostics"
        diagnostics = {}
        for scene, matrix in scenes.items():
            values = np.column_stack([model.predict(matrix, num_threads=2) for model in models["H"]])
            predictions[scene + "__H"] = values
            diagnostics[scene] = base.evaluate_scores(test, values, "test")
        base.write_json(output / "historical_diagnostics.json", diagnostics)
        LOGGER.info("HISTORICAL missing diagnostics complete; training protocol unchanged")
        for name in "ABCD":
            stage = "train_" + name
            folder = output / name
            folder.mkdir()
            models[name] = train_model(name, pairs, data, folder)
            for scene, matrix in scenes.items():
                inputs = spec.feature_view(matrix, name)
                predictions[scene + "__" + name] = np.column_stack([model.predict(inputs, num_threads=2) for model in models[name]])
        np.savez_compressed(output / "predictions.npz", **predictions)
        stage = "evaluation"
        evaluation.analyze(test.reset_index(drop=True), predictions, scenes, output)
        base.write_json(output / "status.json", {"status": "complete", "elapsed_seconds": time.monotonic() - started,
                        "trained_models": 8, "scenarios_including_mask_repeats": len(scenes)})
        LOGGER.info("JOBREC_V4_COMPLETE")
    except Exception as error:
        base.write_json(output / "status.json", {"status": "failed", "stage": stage, "error": str(error)})
        LOGGER.exception("v4 failed at %s; no fallback/retry", stage)
        raise


if __name__ == "__main__":
    main()
