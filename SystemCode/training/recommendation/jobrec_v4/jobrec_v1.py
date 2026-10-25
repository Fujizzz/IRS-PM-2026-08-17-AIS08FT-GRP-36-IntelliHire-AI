"""Responsibilities: run an independent fixed-protocol feasibility study for a dual-head recommendation
model on a Kaggle GPU.
Implementation: download a fixed data release, consolidate consistent duplicate pairs, split by job,
compare text and structured baselines, train a dual-head network on frozen Qwen embeddings, and assess
top-10 qualification reranking.
Related Modules: this historical source does not import or alter AiInterviewer business modules;
Kaggle kernel metadata controls private remote execution.

Declaration Index:
- write_json: atomically write audit records and results.
- bootstrap: install the specified Transformers version, check CUDA, and record the environment.
- download_data: download fixed-version CSVs and record SHA256 hashes.
- consolidate_labels: validate and merge duplicate labels while preserving each task's missing state.
- split_jobs: create a fixed, mutually exclusive job split.
- parse_list: parse Python-style list values from the data.
- profile_texts: construct model inputs without identity or labels.
- structured_features: compute pairwise features with explicit semantics.
- prepare_data: verify entities, labels, distributions, and split separation.
- ranking_metrics: compute ranking and query-coverage metrics on the explicitly labeled candidate pool.
- evaluate_scores: compute AP, AUC, and ranking metrics for both tasks.
- lexical_baseline: fit TF-IDF on training-side text only.
- encode_profiles: encode with a fixed token limit and record truncation rates.
- build_pair_features: combine frozen embeddings with structured inputs.
- DualHeadModel: define shared representation and separate preference/qualification heads.
- DualHeadModel.__init__: create the fixed network layers.
- DualHeadModel.forward: produce two uncalibrated logits.
- train_dual_head: train with label masks and select checkpoints by validation AP only.
- structured_baseline: fit training-side standardization and two logistic regressions.
- rerank_qualification: rerank the semantic-retrieval top 10 for each test job.
- write_report: save readable metrics, boundaries, and configuration.
- main: run stages in dependency order and record/re-raise failures.

Variable Index:
- CONFIG: fixed data release, model revision, split, preprocessing, and training configuration.
- FILES: seven public synthetic data files allowed for download.
- STRUCTURED_NAMES: ordered structured feature names and meanings.
- LOGGER: records stage, size, statistics, and errors without credentials or profile text.

Constraints: CUDA is required; do not switch models, shrink inputs, or retry failed operations. Results
are from a synthetic-data new-job experiment; they do not reproduce the authors' original split, and
unobserved labels are not treated as negative."""

from __future__ import annotations

import ast
import copy
import gc
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import random
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

CONFIG = {
    "protocol": "jobrec-cold-job-feasibility-v1",
    "seed": 42,
    "data_commit": "c742feea3730e8d34ec2e8ae7e58c8c0ee8a53fd",
    "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
    "embedding_revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
    "reranker_model": "Qwen/Qwen3-Reranker-0.6B",
    "reranker_revision": "e61197ed45024b0ed8a2d74b80b4d909f1255473",
    "transformers_version": "4.56.2",
    "job_split_counts": [70, 15, 15],
    "embedding_max_length": 384,
    "embedding_batch_size": 8,
    "embedding_dtype": "float16",
    "attention": "sdpa",
    "job_instruction": "Given a job description, retrieve candidates qualified for the job.",
    "epochs": 12,
    "training_batch_size": 512,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "hidden_sizes": [128, 32],
    "dropout": 0.1,
    "tfidf_max_features": 20000,
    "tfidf_min_df": 2,
    "logistic_max_iter": 2000,
    "rerank_top_k": 10,
    "rerank_batch_size": 4,
    "rerank_query_tokens": 160,
    "rerank_document_tokens": 320,
    "rerank_max_length": 768,
    "ranking_k": [5, 10],
    "selection_metric": "mean_validation_average_precision_over_two_observed_tasks",
}
FILES = (
    "Final_users.csv", "Final_items.csv", "train.csv", "val.csv", "test.csv",
    "qual_train.csv", "qual_test.csv",
)
STRUCTURED_NAMES = (
    "skill_coverage", "interest_matches_industry", "major_accepted", "work_mode_matches",
    "gpa_margin", "experience_months_margin", "academic_level_margin",
    "weekly_hours_margin", "commitment_months_margin", "publications_margin",
    "summer_commitment",
)
LOGGER = logging.getLogger("jobrec")


def write_json(path: Path, value: object) -> None:
    """Functionality: atomically save a JSON value.
    Inputs: destination path and JSON-serializable value. Outputs: none; writes the file.
    Logic: write a temporary sibling file, then replace the destination.
    Constraints: reject nonfinite values; propagate write failures."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def bootstrap(output: Path) -> None:
    """Functionality: install the fixed dependencies and verify the GPU environment.
    Inputs: output directory. Outputs: none.
    Logic: install the pinned Transformers version, set seeds, and record platform package versions.
    Constraints: fail when CUDA is unavailable; do not change hardware or parameters. Retain and record
    Kaggle-preinstalled Torch/NumPy; this is not a fully locked environment."""
    os.environ["HF_HOME"] = "/tmp/jobrec_hf"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    subprocess.run([
        sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check",
        "transformers==" + CONFIG["transformers_version"],
    ], check=True)
    if not torch.cuda.is_available():
        raise RuntimeError("Protocol requires CUDA; refusing a CPU substitution.")
    random.seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])
    torch.cuda.manual_seed_all(CONFIG["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(2)
    environment = {
        "python": sys.version, "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in (
            "torch", "numpy", "pandas", "scikit-learn", "transformers", "tokenizers",
            "huggingface-hub", "safetensors",
        )},
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "used_device": "cuda:0", "cuda": torch.version.cuda,
        "note": "Seeds fixed; bitwise reproducibility across GPU/library versions is not guaranteed.",
    }
    write_json(output / "environment.json", environment)
    LOGGER.info("Environment ready: GPUs=%s", environment["gpu_names"])


def download_data(output: Path) -> dict[str, pd.DataFrame]:
    """Functionality: obtain the seven CSVs from the fixed data release.
    Inputs: output directory. Outputs: mapping from file names to DataFrames.
    Logic: request public URLs and record each file's hash and row count.
    Constraints: no retries; do not write credentials or a duplicate of raw profile text."""
    import io

    frames, manifest = {}, {}
    base = "https://raw.githubusercontent.com/brycekan123/DualOptimization_jobrec/"
    for filename in FILES:
        url = base + CONFIG["data_commit"] + "/dataset/" + filename
        with urllib.request.urlopen(url, timeout=60) as response:
            raw = response.read()
        frame = pd.read_csv(io.BytesIO(raw))
        frames[filename] = frame
        manifest[filename] = {"url": url, "sha256": hashlib.sha256(raw).hexdigest(), "rows": len(frame)}
        LOGGER.info("Downloaded %s: %d rows", filename, len(frame))
    write_json(output / "data_manifest.json", manifest)
    return frames


def consolidate_labels(frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict]:
    """Functionality: construct the unique pair table and audit raw duplicates.
    Inputs: source CSV DataFrames. Outputs: merged pair table and duplicate audit.
    Logic: retain only consistent labels for the same task/pair; outer-join tasks to preserve NaN masks.
    Constraints: conflicting or nonbinary labels raise an error; ignore source scores; never fill missing
    qualification labels with zero."""
    tables, audit = [], {}
    for task, names in (("pref", ("train.csv", "val.csv", "test.csv")),
                        ("qual", ("qual_train.csv", "qual_test.csv"))):
        label = "label_" + task
        combined = pd.concat([frames[name][["user", "item", label]] for name in names], ignore_index=True)
        if combined.isna().any().any() or not set(combined[label].unique()).issubset({0, 1}):
            raise ValueError(f"Invalid identifiers or binary labels for {task}")
        if (combined.groupby(["user", "item"])[label].nunique() > 1).any():
            raise ValueError(f"Conflicting duplicate labels for {task}")
        unique = combined.drop_duplicates(["user", "item"]).copy()
        audit[task] = {"source_rows": len(combined), "unique_pairs": len(unique),
                       "consistent_duplicates_collapsed": len(combined) - len(unique)}
        tables.append(unique)
    merged = tables[0].merge(tables[1], on=["user", "item"], how="outer", validate="one_to_one")
    return merged.sort_values(["user", "item"]).reset_index(drop=True), audit


def split_jobs(job_ids: list[str]) -> dict[str, str]:
    """Functionality: assign a fixed cold-job split.
    Inputs: job IDs. Outputs: mapping from job IDs to split names.
    Logic: sort IDs, then apply an independent seeded permutation.
    Constraints: require exactly 100 distinct jobs and fixed 70/15/15 proportions."""
    ids = sorted(job_ids)
    if len(ids) != 100 or len(set(ids)) != 100:
        raise ValueError("Expected exactly 100 distinct jobs.")
    shuffled = np.random.default_rng(CONFIG["seed"]).permutation(ids)
    return {str(job): ("train" if i < 70 else "validation" if i < 85 else "test")
            for i, job in enumerate(shuffled)}


def parse_list(value: str) -> list[str]:
    """Functionality: parse one source-data list value.
    Inputs: Python list text. Outputs: list of strings.
    Logic: use literal_eval and validate the parsed type.
    Constraints: malformed values raise an error; do not guess alternate separators."""
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise ValueError("Expected a list of strings in profile.")
    return parsed


def profile_texts(users: pd.DataFrame, jobs: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Functionality: construct model text for candidate profiles and job queries.
    Inputs: complete profile tables. Outputs: candidate texts and job-query texts.
    Logic: place skills/interests/education/experience first and synthetic descriptions afterward; use a
    fixed qualification-retrieval instruction for jobs.
    Constraints: exclude IDs, match labels, source scores, and fictional interview evidence; later truncate
    to the fixed token limit."""
    candidate_texts = [
        f"Skills: {', '.join(parse_list(row.skills))}. Interests: {', '.join(parse_list(row.interests))}. "
        f"Education: {row.academic_level}, {row.masters_major}, {row.undergrad_major}. "
        f"Experience months: {row.months_experience}. Profile: {row.student_llm_output}"
        for row in users.itertuples()
    ]
    job_texts = [
        f"Instruct: {CONFIG['job_instruction']}\nQuery: "
        f"Industry: {row.industry}. Skills: {', '.join(parse_list(row.required_skills))}. "
        f"Education: {row.min_academic_level}. Experience months: {row.min_months_experience}. "
        f"Description: {row.job_llm_output}"
        for row in jobs.itertuples()
    ]
    return candidate_texts, job_texts


def structured_features(pairs: pd.DataFrame, users: pd.DataFrame, jobs: pd.DataFrame) -> np.ndarray:
    """Functionality: calculate eleven explicit match features.
    Inputs: pair table and profile tables. Outputs: float32 feature matrix.
    Logic: compute skill coverage, preference/major/work-mode matches, and numeric margins.
    Constraints: qualification labels may be generated from similar rules, so this baseline measures
    learnability on synthetic tasks only. It contains no demographic attributes or nonexistent interview
    scores."""
    user_map = users.set_index("student_id").to_dict("index")
    job_map = jobs.set_index("job_id").to_dict("index")
    levels = {"UG1": 1, "UG2": 2, "UG3": 3, "UG4": 4, "MS1": 5, "MS2": 6,
              "PhD1": 7, "PhD2": 8, "PhD3": 9, "PhD4": 10, "PhD5": 11}
    result = []
    for row in pairs.itertuples():
        user, job = user_map[row.user], job_map[row.item]
        skills, required = set(parse_list(user["skills"])), set(parse_list(job["required_skills"]))
        if not required:
            raise ValueError("Empty required skill list is outside the frozen protocol.")
        majors = {user["masters_major"], user["second_major"], user["undergrad_major"]}
        features = [len(skills & required) / len(required),
                    float(job["industry"] in parse_list(user["interests"])),
                    float(bool(majors & set(parse_list(job["acceptable_majors"])))),
                    float(user["in_person_commitment"] == job["job_in_person_commitment"]),
                    float(user["gpa"]) - float(job["min_gpa"]),
                    float(user["months_experience"]) - float(job["min_months_experience"]),
                    float(levels[user["academic_level"]] - levels[job["min_academic_level"]]),
                    float(user["hours_per_week"]) - float(job["min_hours_per_week"]),
                    float(user["length_of_commitment"]) - float(job["min_length_of_commitment"]),
                    float(user["num_publications"]) - float(job["min_num_publications"]),
                    float(str(user["commit_to_summer"]).lower() == "true")]
        result.append(features)
    matrix = np.asarray(result, dtype=np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError("Non-finite structured features.")
    return matrix


def prepare_data(frames: dict[str, pd.DataFrame], output: Path) -> tuple:
    """Functionality: map entities and audit the split.
    Inputs: source tables and output path. Outputs: all prepared training objects.
    Logic: merge pairs, split by job, and verify that each task/split has positive and negative labels.
    Constraints: candidates may cross splits, so this is a cold-job—not cold-candidate—study. Candidate
    pools contain only source pairs with explicit labels."""
    users = frames["Final_users.csv"].sort_values("student_id").reset_index(drop=True)
    jobs = frames["Final_items.csv"].sort_values("job_id").reset_index(drop=True)
    if users.student_id.duplicated().any() or jobs.job_id.duplicated().any():
        raise ValueError("Entity IDs must be unique.")
    pairs, audit = consolidate_labels(frames)
    split_map = split_jobs(jobs.job_id.tolist())
    pairs["split"] = pairs.item.map(split_map)
    pairs["u_idx"] = pairs.user.map(dict(zip(users.student_id, users.index)))
    pairs["j_idx"] = pairs.item.map(dict(zip(jobs.job_id, jobs.index)))
    if pairs[["split", "u_idx", "j_idx"]].isna().any().any():
        raise ValueError("Unknown entity in label table.")
    for split in ("train", "validation", "test"):
        audit[split] = {}
        for task in ("pref", "qual"):
            observed = pairs.loc[pairs.split.eq(split), "label_" + task].dropna()
            if set(observed.unique()) != {0, 1}:
                raise ValueError(f"Both classes required: {split}/{task}")
            audit[split][task] = {"pairs": len(observed), "positives": int(observed.sum())}
    audit["protocol_note"] = "Disjoint jobs; shared candidates; observed labeled candidate pools only."
    audit["structured_feature_names"] = STRUCTURED_NAMES
    write_json(output / "split_manifest.json", split_map)
    write_json(output / "data_audit.json", audit)
    pairs.to_csv(output / "pair_splits.csv", index=False)
    texts = profile_texts(users, jobs)
    features = structured_features(pairs, users, jobs)
    LOGGER.info("Data audit: %s", json.dumps(audit))
    return pairs, users, jobs, texts, features


def ranking_metrics(frame: pd.DataFrame, scores: np.ndarray, task: str) -> dict:
    """Functionality: compute macro-averaged ranking metrics on explicitly labeled candidate pools.
    Inputs: observed-label rows, scores, and task. Outputs: metrics dictionary.
    Logic: group preference rows by candidate and qualification rows by job; exclude groups without
    positives from the mean and report them separately; break ties by stable input order.
    Constraints: this is not full-corpus retrieval; nDCG/Recall denominators use only labeled candidates."""
    scored = frame.assign(pred=np.asarray(scores))
    group_column = "user" if task == "pref" else "item"
    label_column = "label_" + task
    per_query, pool_sizes, all_negative = [], [], 0
    for _, group in scored.groupby(group_column, sort=True):
        labels = group[label_column].to_numpy()
        positives = int(labels.sum())
        if positives == 0:
            all_negative += 1
            continue
        ranked = group.sort_values("pred", ascending=False, kind="stable")[label_column].to_numpy()
        values = {}
        for k in CONFIG["ranking_k"]:
            limit = min(k, len(ranked))
            discount = 1 / np.log2(np.arange(limit) + 2)
            ideal = (1 / np.log2(np.arange(min(positives, limit)) + 2)).sum()
            values[f"ndcg@{k}"] = float((ranked[:limit] * discount).sum() / ideal)
            values[f"recall@{k}"] = float(ranked[:limit].sum() / positives)
        per_query.append(values)
        pool_sizes.append(len(group))
    if not per_query:
        raise ValueError(f"No evaluable positive queries for {task}")
    return {**{name: float(np.mean([row[name] for row in per_query])) for name in per_query[0]},
            "positive_queries": len(per_query), "all_negative_queries_excluded": all_negative,
            "median_labeled_pool": float(np.median(pool_sizes))}


def evaluate_scores(pairs: pd.DataFrame, scores: np.ndarray, split: str) -> dict:
    """Functionality: evaluate both tasks on one data split.
    Inputs: pair table, N-by-2 scores, and split name. Outputs: per-task metrics.
    Logic: use explicit labels only.
    Constraints: sigmoid/ranking scores are not calibrated business probabilities."""
    result = {}
    for column, task in enumerate(("pref", "qual")):
        selected = pairs.split.eq(split) & pairs["label_" + task].notna()
        frame = pairs.loc[selected]
        labels = frame["label_" + task].to_numpy()
        values = scores[selected.to_numpy(), column]
        result[task] = {"average_precision": float(average_precision_score(labels, values)),
                        "roc_auc": float(roc_auc_score(labels, values)),
                        "positive_prevalence": float(labels.mean()), "pairs": len(frame),
                        **ranking_metrics(frame, values, task)}
    return result


def lexical_baseline(pairs: pd.DataFrame, texts: tuple, output: Path) -> np.ndarray:
    """Functionality: calculate TF-IDF pairwise cosine scores.
    Inputs: pairs, text representations, and output directory. Outputs: shared semantic scores for N pairs.
    Logic: fit vocabulary only on entities appearing in training pairs; transform validation/test only.
    Constraints: these scores do not learn bilateral preference; record vocabulary size without serializing
    the profile vocabulary."""
    candidate_texts, job_texts = texts
    train = pairs.loc[pairs.split.eq("train")]
    corpus = [candidate_texts[i] for i in sorted(train.u_idx.unique())]
    corpus += [job_texts[i] for i in sorted(train.j_idx.unique())]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=CONFIG["tfidf_min_df"],
                                 max_features=CONFIG["tfidf_max_features"], dtype=np.float32)
    vectorizer.fit(corpus)
    u, j = vectorizer.transform(candidate_texts), vectorizer.transform(job_texts)
    values = np.asarray(u[pairs.u_idx].multiply(j[pairs.j_idx]).sum(axis=1)).ravel()
    write_json(output / "lexical_audit.json", {"vocabulary_size": len(vectorizer.vocabulary_),
               "training_entities_only": True})
    return np.column_stack([values, values])


def encode_profiles(texts: tuple, output: Path) -> tuple[np.ndarray, np.ndarray]:
    """Functionality: produce frozen Qwen representations.
    Inputs: candidate/job text and output directory. Outputs: normalized embeddings for both sides.
    Logic: use the fixed revision, left padding, last-valid-token pooling, FP16, SDPA, and cuda:0.
    Constraints: retain the 384-token limit and fail rather than changing it after memory errors; save only
    embeddings and truncation statistics, never profile text."""
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(CONFIG["embedding_model"],
                 revision=CONFIG["embedding_revision"], padding_side="left")
    model = AutoModel.from_pretrained(CONFIG["embedding_model"],
             revision=CONFIG["embedding_revision"], torch_dtype=torch.float16,
             attn_implementation=CONFIG["attention"]).to("cuda:0").eval()
    all_texts = texts[0] + texts[1]
    chunks, lengths, started = [], [], time.monotonic()
    batch_size = CONFIG["embedding_batch_size"]
    for offset in range(0, len(all_texts), batch_size):
        batch = all_texts[offset:offset + batch_size]
        raw_tokens = tokenizer(batch, truncation=False, add_special_tokens=True)["input_ids"]
        lengths.extend(len(row) for row in raw_tokens)
        inputs = tokenizer(batch, padding=True, truncation=True,
                  max_length=CONFIG["embedding_max_length"], return_tensors="pt").to("cuda:0")
        with torch.inference_mode():
            hidden = model(**inputs).last_hidden_state[:, -1]
            encoded = torch.nn.functional.normalize(hidden.float(), dim=-1)
        chunks.append(encoded.cpu().numpy())
        if offset % (batch_size * 25) == 0:
            LOGGER.info("Embedding %d/%d elapsed=%.1fs", offset + len(batch), len(all_texts), time.monotonic() - started)
    embeddings = np.concatenate(chunks)
    if not np.isfinite(embeddings).all():
        raise ValueError("Non-finite embeddings")
    audit = {"entities": len(all_texts), "dimension": embeddings.shape[1],
             "elapsed_seconds": time.monotonic() - started,
             "truncated_entities": int(np.sum(np.asarray(lengths) > CONFIG["embedding_max_length"])),
             "median_original_tokens": float(np.median(lengths)),
             "semantic_baseline": "job-query instruction; same qualification-oriented cosine in both directions"}
    write_json(output / "embedding_audit.json", audit)
    np.savez_compressed(output / "embeddings.npz", candidates=embeddings[:len(texts[0])], jobs=embeddings[len(texts[0]):])
    del model, inputs, hidden, encoded
    gc.collect()
    torch.cuda.empty_cache()
    LOGGER.info("Embedding completed: %s", audit)
    return embeddings[:len(texts[0])], embeddings[len(texts[0]):]


def build_pair_features(pairs: pd.DataFrame, embeddings: tuple, structured: np.ndarray,
                        output: Path) -> tuple[np.ndarray, np.ndarray]:
    """Functionality: combine frozen embeddings and structured pair features.
    Inputs: pair table, embeddings, features, and output directory. Outputs: network inputs and cosine scores.
    Logic: concatenate absolute embedding differences, elementwise products, and training-side
    standardized structured features.
    Constraints: fit standardization on training rows only."""
    u, j = embeddings[0][pairs.u_idx], embeddings[1][pairs.j_idx]
    scaler = StandardScaler().fit(structured[pairs.split.eq("train")])
    features = np.concatenate([np.abs(u - j), u * j, scaler.transform(structured)], axis=1).astype(np.float32)
    cosine = (u * j).sum(axis=1)
    write_json(output / "feature_scaler.json", {"names": STRUCTURED_NAMES,
               "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()})
    return features, np.column_stack([cosine, cosine])


class DualHeadModel(torch.nn.Module):
    """Functionality: define the shared-representation preference/qualification network.
    Logic: use an MLP trunk followed by one linear output per task.
    Constraints: the heads use independent observed-label masks; instance state contains trained weights,
    not entity IDs."""

    def __init__(self, feature_size: int):
        """Functionality: initialize the network.
        Inputs: feature dimension. Outputs: a new model instance.
        Logic: create fixed 128- and 32-unit hidden layers with dropout.
        Constraints: layer parameters must match CONFIG."""
        super().__init__()
        first, second = CONFIG["hidden_sizes"]
        self.trunk = torch.nn.Sequential(torch.nn.Linear(feature_size, first), torch.nn.ReLU(),
                     torch.nn.Dropout(CONFIG["dropout"]), torch.nn.Linear(first, second), torch.nn.ReLU())
        self.heads = torch.nn.ModuleList([torch.nn.Linear(second, 1), torch.nn.Linear(second, 1)])

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Functionality: produce preference and qualification logits.
        Inputs: B-by-D feature tensor. Outputs: B-by-2 logits.
        Logic: apply the shared trunk and concatenate the two linear heads.
        Constraints: the inference caller is responsible for eval mode and sigmoid conversion."""
        hidden = self.trunk(features)
        return torch.cat([head(hidden) for head in self.heads], dim=1)


def train_dual_head(pairs: pd.DataFrame, features: np.ndarray, output: Path) -> np.ndarray:
    """Functionality: train the dual-head model and score every pair.
    Inputs: pair table, features, and output directory. Outputs: N-by-2 scores.
    Logic: mask missing labels, use training-side positive-class weights, and select the best of 12 epochs
    by mean validation AP across both tasks.
    Constraints: never select from test results; no early stopping or automatic parameter changes; compute
    test metrics only after training is complete."""
    model = DualHeadModel(features.shape[1]).to("cuda:0")
    x = torch.from_numpy(features).to("cuda:0")
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
    criterion = torch.nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor(weights, device="cuda:0"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    best_score, best_state, history = -1.0, None, []
    rng = np.random.default_rng(CONFIG["seed"])
    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        losses, order = [], rng.permutation(train_ids)
        for start in range(0, len(order), CONFIG["training_batch_size"]):
            ids = order[start:start + CONFIG["training_batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            element_loss = criterion(model(x[ids]), y[ids])
            task_losses = [(element_loss[:, c] * mask[ids, c]).sum() / mask[ids, c].sum()
                           for c in range(2) if mask[ids, c].any()]
            loss = torch.stack(task_losses).mean()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        model.eval()
        with torch.inference_mode():
            val_scores = torch.sigmoid(model(x[val_ids])).cpu().numpy()
        aps = [float(average_precision_score(targets[val_ids, c][observed[val_ids, c]],
               val_scores[:, c][observed[val_ids, c]])) for c in range(2)]
        validation_mean = float(np.mean(aps))
        row = {"epoch": epoch, "mean_batch_loss": float(np.mean(losses)), "validation_ap": aps}
        history.append(row)
        LOGGER.info("Training epoch: %s", row)
        if validation_mean > best_score:
            best_score = validation_mean
            best_state = copy.deepcopy(model.state_dict())
            selected_epoch = epoch
    if best_state is None:
        raise RuntimeError("No finite validation checkpoint")
    model.load_state_dict(best_state)
    torch.save({"state_dict": {k: v.cpu() for k, v in best_state.items()},
                "feature_size": features.shape[1], "config": CONFIG}, output / "dual_head.pt")
    with torch.inference_mode():
        scores = torch.sigmoid(model(x)).cpu().numpy()
    write_json(output / "training_history.json", {"epochs": history, "selected_epoch": selected_epoch,
               "selection_validation_mean_ap": best_score, "positive_weights": weights})
    del model, x, y, mask, optimizer, best_state
    gc.collect()
    torch.cuda.empty_cache()
    return scores


def structured_baseline(pairs: pd.DataFrame, features: np.ndarray, output: Path) -> np.ndarray:
    """Functionality: fit structured logistic-regression baselines.
    Inputs: pairs, raw 11-dimensional features, and output directory. Outputs: N-by-2 scores.
    Logic: fit scaling and logistic regression for each task using only that task's training labels.
    Constraints: no label propagation between tasks."""
    predictions, records = [], {}
    for task in ("pref", "qual"):
        observed = pairs.split.eq("train") & pairs["label_" + task].notna()
        scaler = StandardScaler().fit(features[observed])
        x = scaler.transform(features)
        model = LogisticRegression(class_weight="balanced", random_state=CONFIG["seed"],
                                    max_iter=CONFIG["logistic_max_iter"])
        model.fit(x[observed], pairs.loc[observed, "label_" + task])
        if int(model.n_iter_[0]) >= CONFIG["logistic_max_iter"]:
            raise RuntimeError(f"Logistic regression did not converge: {task}")
        predictions.append(model.predict_proba(x)[:, 1])
        records[task] = {"names": STRUCTURED_NAMES, "coef": model.coef_[0].tolist(),
                         "intercept": model.intercept_.tolist(), "iterations": model.n_iter_.tolist()}
    write_json(output / "structured_logistic.json", records)
    return np.column_stack(predictions)


def rerank_qualification(pairs: pd.DataFrame, texts: tuple, cosine: np.ndarray, output: Path) -> dict:
    """Functionality: run two-stage reranking for qualification.
    Inputs: pairs, text, cosine scores, and output directory. Outputs: metrics on the fixed test candidate pool.
    Logic: retrieve the top 10 by semantic score per job, rerank the prefix with Qwen yes/no logits, and
    preserve the original order for the tail.
    Constraints: query/document limits are fixed at 160/320 tokens; this is not full-corpus retrieval; do
    not report a pseudo-probability AUC or modify the candidate pool."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    frame = pairs.loc[pairs.split.eq("test") & pairs.label_qual.notna()].copy()
    frame["semantic"] = cosine[frame.index, 1]
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["reranker_model"],
                 revision=CONFIG["reranker_revision"], padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(CONFIG["reranker_model"],
             revision=CONFIG["reranker_revision"], torch_dtype=torch.float16,
             attn_implementation=CONFIG["attention"]).to("cuda:0").eval()
    prefix = ('<|im_start|>system\nJudge whether the Document meets the requirements based on the Query '
              'and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n')
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    prefix_ids, suffix_ids = tokenizer.encode(prefix, add_special_tokens=False), tokenizer.encode(suffix, add_special_tokens=False)
    selected, payloads = [], []
    ranked_groups = []
    for _, group in frame.groupby("item", sort=True):
        ranked = group.sort_values("semantic", ascending=False, kind="stable")
        ranked_groups.append(ranked.index.tolist())
        for row in ranked.head(CONFIG["rerank_top_k"]).itertuples():
            query_ids = tokenizer.encode(texts[1][row.j_idx], add_special_tokens=False)[:CONFIG["rerank_query_tokens"]]
            doc_ids = tokenizer.encode(texts[0][row.u_idx], add_special_tokens=False)[:CONFIG["rerank_document_tokens"]]
            body = f"<Instruct>: {CONFIG['job_instruction']}\n<Query>: {tokenizer.decode(query_ids)}\n<Document>: {tokenizer.decode(doc_ids)}"
            ids = prefix_ids + tokenizer.encode(body, add_special_tokens=False) + suffix_ids
            if len(ids) > CONFIG["rerank_max_length"]:
                raise ValueError("Reranker formatted sequence exceeds frozen limit")
            selected.append(row.Index)
            payloads.append(ids)
    values, started = [], time.monotonic()
    yes, no = tokenizer.convert_tokens_to_ids("yes"), tokenizer.convert_tokens_to_ids("no")
    for start in range(0, len(payloads), CONFIG["rerank_batch_size"]):
        encoded = tokenizer.pad({"input_ids": payloads[start:start + CONFIG["rerank_batch_size"]]},
                                padding=True, return_tensors="pt").to("cuda:0")
        with torch.inference_mode():
            logits = model(**encoded, logits_to_keep=1).logits[:, -1, :]
            scores = torch.softmax(logits[:, [no, yes]].float(), dim=-1)[:, 1]
        values.extend(scores.cpu().tolist())
        if start % 40 == 0:
            LOGGER.info("Reranking %d/%d", start + len(scores), len(payloads))
    by_index = dict(zip(selected, values))
    final = pd.Series(index=frame.index, dtype=float)
    for indexes in ranked_groups:
        head = indexes[:CONFIG["rerank_top_k"]]
        reordered = sorted(head, key=by_index.__getitem__, reverse=True) + indexes[CONFIG["rerank_top_k"]:]
        for rank, index in enumerate(reordered):
            final.loc[index] = len(reordered) - rank
    result = {"qual": ranking_metrics(frame, final.loc[frame.index].to_numpy(), "qual"),
              "reranked_pairs": len(selected), "elapsed_seconds": time.monotonic() - started,
              "scope": "test qualification labeled pools; semantic top10 prefix only"}
    write_json(output / "reranker_audit.json", result)
    pd.DataFrame({"pair_index": selected, "reranker_yes_score": values}).to_csv(output / "reranker_scores.csv", index=False)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def write_report(results: dict, output: Path) -> None:
    """Functionality: write the readable experiment report.
    Inputs: metric results and output directory. Outputs: Markdown report.
    Logic: report each task and candidate-pool metric separately.
    Constraints: do not describe synthetic-label results as real hiring or interview gains."""
    lines = ["# JobRec cold-job feasibility v1", "", "Completed on Kaggle; frozen Qwen embeddings + masked dual-head MLP.",
             "", "| Model | Task | AP | AUC | NDCG@5 | Recall@5 | Queries | Median labeled pool |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for name, metrics in results["models"].items():
        for task, row in metrics.items():
            lines.append(f"| {name} | {task} | {row['average_precision']:.4f} | {row['roc_auc']:.4f} | "
                         f"{row['ndcg@5']:.4f} | {row['recall@5']:.4f} | {row['positive_queries']} | {row['median_labeled_pool']:.1f} |")
    lines += ["", "## Reranker", "", json.dumps(results["reranker"], indent=2), "", "## Limits", "",
              "- Fully synthetic, rule-derived supervision; no real hiring outcome validation.",
              "- Jobs split 70/15/15; candidates may recur. This is a new cold-job protocol, not paper reproduction.",
              "- Source repeated labels collapsed only after consistency checks; unknown labels stay masked.",
              "- Rankings use observed labeled pools, not the full catalog; both task pools differ.",
              "- Qualification labels are conditional on the source selection process and are very sparse.",
              "- No interview features exist in this dataset; interview benefit has not been tested.",
              "- Text length, seed, checkpoints and training settings were fixed before execution.",
              "- AP prevalence baselines and evaluated query counts are in metrics.json.",
              "- Dataset repository license was not explicit; kernel remains private."]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Functionality: execute the full experiment.
    Inputs: Kaggle GPU/network environment. Outputs: artifacts under `/kaggle/working/jobrec_v1`.
    Logic: save audit data after each stage, then save models, predictions, and metrics.
    Constraints: record the failing stage and re-raise any exception; do not rerun, degrade, or change
    parameters automatically. Preserve evidence from completed stages."""
    output = Path("/kaggle/working/jobrec_v1")
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(output / "execution.log", encoding="utf-8")])
    stage, started = "bootstrap", time.monotonic()
    write_json(output / "config.json", CONFIG)
    try:
        bootstrap(output)
        stage = "download_and_audit"
        frames = download_data(output)
        pairs, users, jobs, texts, structured = prepare_data(frames, output)
        stage = "baselines"
        lexical = lexical_baseline(pairs, texts, output)
        logistic = structured_baseline(pairs, structured, output)
        stage = "frozen_qwen_embedding"
        embeddings = encode_profiles(texts, output)
        features, cosine = build_pair_features(pairs, embeddings, structured, output)
        stage = "dual_head_training"
        neural = train_dual_head(pairs, features, output)
        predictions = {"tfidf": lexical, "structured_logistic": logistic,
                       "qwen_embedding": cosine, "dual_head_mlp": neural}
        results = {"protocol": CONFIG["protocol"],
                   "models": {name: evaluate_scores(pairs, score, "test") for name, score in predictions.items()}}
        write_json(output / "metrics_before_reranker.json", results)
        for name, values in predictions.items():
            pairs[name + "_pref"] = values[:, 0]
            pairs[name + "_qual"] = values[:, 1]
        pairs.loc[pairs.split.eq("test")].to_csv(output / "test_predictions.csv", index=False)
        stage = "qualification_reranker"
        results["reranker"] = rerank_qualification(pairs, texts, cosine, output)
        results["total_seconds"] = time.monotonic() - started
        write_json(output / "metrics.json", results)
        write_report(results, output)
        write_json(output / "status.json", {"status": "complete", "elapsed_seconds": results["total_seconds"]})
        LOGGER.info("EXPERIMENT_COMPLETE %s", json.dumps(results))
    except Exception as error:
        write_json(output / "status.json", {"status": "failed", "stage": stage,
                   "error_type": type(error).__name__, "error": str(error), "elapsed_seconds": time.monotonic() - started})
        LOGGER.exception("Experiment failed at stage=%s; no automatic fallback", stage)
        raise


if __name__ == "__main__":
    main()
