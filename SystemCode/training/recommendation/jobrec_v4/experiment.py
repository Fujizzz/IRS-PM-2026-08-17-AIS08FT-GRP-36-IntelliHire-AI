"""Responsibilities: execute the approved v4 missing-profile training study and verify its historical baseline.
Implementation: verify fixed parent-artifact hashes, rebuild masked-profile features, fit eight rankers,
and save predictions for a unified analysis.
Related Modules: jobrec_v1/jobrec_v2 provide frozen historical helpers; jobrec_missing defines missingness;
jobrec_missing_eval analyzes outputs.

Declaration Index:
- verify_history: verify the identities of v3 rankers and recorded evaluation artifacts.
- prepare_scenes: rebuild registered missing-profile scenes from hidden source fields.
- train_model: fit one registered pair of preference/qualification rankers.
- main: execute the staged data, training, and evaluation workflow.

Variable Index:
- LOGGER: stage-level logging without credentials or profile text; experiment settings come from
  spec.PROTOCOL and spec.RANKER_CONFIG.

Constraints: preserve historical models, splits, labels, and tree parameters; each failed stage is
recorded and re-raised, with no retry or fallback."""

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
    """Functionality: verify the identity of historical v3 rankers and result files.
    Inputs: parent artifact directory. Outputs: none.
    Logic: compare each file's SHA256 with the frozen protocol manifest.
    Constraints: any mismatch raises an error; a newer file with the same name is not accepted."""
    for name, expected in spec.HISTORICAL.items():
        if hashlib.sha256((folder / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"v3 historical artifact changed: {name}")
    LOGGER.info("Historical v3 artifacts verified")


def prepare_scenes(pairs: pd.DataFrame, users: pd.DataFrame, jobs: pd.DataFrame,
                   clean: np.ndarray, output: Path) -> dict:
    """Functionality: construct the fixed missing-profile scenarios.
    Inputs: pair/user/job tables, the complete reference feature matrix, and the output directory.
    Outputs: a mapping from scenario keys to feature matrices for all pairs.
    Logic: mask entity fields before recomputing features, then save entity IDs, masks, and audit summaries.
    Constraints: the clean reconstruction must match element by element; original profiles are not written
    as scenario output; training masks use a separate namespace; missing feature values remain NaN and do
    not fill missing labels."""
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
    """Functionality: fit one preregistered bidirectional comparison variant.
    Inputs: variant name, pair table, scenario feature matrices, and model output directory.
    Outputs: the two selected, reloaded LightGBM Boosters.
    Logic: A uses complete profiles; B/C/D share the fixed mixed-profile training input. Group rows by
    query for each task and select from the 20-to-200 tree grid using the equal mean AP over three
    validation scenarios.
    Constraints: fit on train only and select on validation only; do not duplicate pairs; resolve ties in
    favor of the earlier iteration; save and reload the chosen model."""
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
    """Functionality: run the complete v4 experiment.
    Inputs: mounted Kaggle parent kernels, network access, and the approved fixed protocol.
    Outputs: the `jobrec_v4` artifacts under `/kaggle/working`.
    Logic: verify inputs, initialize the environment, reconstruct data and the clean baseline, diagnose
    missingness, fit eight rankers, and evaluate; log each stage and persist failure status.
    Constraints: retain the historical environment setup. Trees train on CPU with two threads; do not
    claim GPU training. This study does not train text or interview models."""
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
