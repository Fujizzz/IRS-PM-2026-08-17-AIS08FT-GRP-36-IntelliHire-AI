"""职责：在 Kaggle GPU 上运行招聘双头模型的独立、固定协议可行性实验。
实现：下载固定提交的数据；合并一致重复配对；按岗位隔离；比较文本与结构化基线，
      冻结 Qwen 向量后训练双头网络，并评估资格方向的 top-10 精排。
关联：不导入或改变 AiInterviewer 业务模块；kernel-metadata.json 负责私有远端执行。
目录：
- write_json：原子写入审计与结果。
- bootstrap：安装指定 Transformers 版本，检查 CUDA，记录环境。
- download_data：下载固定版本 CSV 并记录 SHA256。
- consolidate_labels：验证并合并重复标签，保留各任务缺失状态。
- split_jobs：生成固定、互斥的岗位划分。
- parse_list：解析数据中的 Python 字符串列表。
- profile_texts：生成无身份、无标签的模型输入。
- structured_features：计算有明确语义的配对特征。
- prepare_data：核对实体、标签、分布及跨划分不重叠。
- ranking_metrics：在明确标注的候选池计算排名指标和查询覆盖。
- evaluate_scores：计算每任务 AP、AUC 与排名指标。
- lexical_baseline：仅用训练侧语料拟合 TF-IDF。
- encode_profiles：按固定上限编码并记录截断比例。
- build_pair_features：组合冻结向量与结构化输入。
- DualHeadModel：共享表示和偏好/资格两个预测头。
- DualHeadModel.__init__：创建固定网络层。
- DualHeadModel.forward：产生两个未校准 logits。
- train_dual_head：以标签掩码训练并仅按验证 AP 选检查点。
- structured_baseline：拟合训练侧标准化与两个逻辑回归。
- rerank_qualification：对测试岗位语义召回的 top-10 做精排。
- write_report：保存可读指标表、边界及配置。
- main：按依赖顺序执行实验，失败时记录阶段和堆栈并重新抛出。
变量索引：
- CONFIG：冻结的数据提交、模型修订、划分、预处理与训练配置。
- FILES：允许下载的七个公开合成数据文件。
- STRUCTURED_NAMES：结构化特征的顺序与含义。
- LOGGER：仅记录阶段、尺寸、统计与错误，不记录凭据和档案正文。
约束：要求 CUDA；不自动换模型、减小输入或重试失败步骤。结果属于合成数据上的
      新岗位实验；不复现作者原始划分，不把未观测标签当负例。
"""

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
    """功能：原子保存 JSON；输入 path/value；输出无，写文件。
    逻辑：同目录临时文件后 replace；约束：拒绝非有限数，写入失败向调用方传播。
    """
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def bootstrap(output: Path) -> None:
    """功能：准备指定依赖并检查 GPU；输入输出目录；输出无。
    逻辑：显式安装固定 Transformers、设种子并记录平台包版本；约束：无 GPU 即失败，
    不更换硬件或参数。Kaggle 预装 torch/numpy 等保留并记录，不能视为完整环境锁定。
    """
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
    """功能：取得固定提交的七份 CSV；输入输出目录；输出文件名到 DataFrame。
    逻辑：请求公开 URL，逐文件计算哈希与行数；约束：无重试、不写凭据和原始档案副本。
    """
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
    """功能：构造唯一配对表及原始重复审计；输入 CSV 表；输出合并表/审计。
    逻辑：相同任务同一配对只保留一致标签；不同任务 outer join 保留 NaN 掩码。
    约束：冲突或非二元标签直接报错；不使用源 score、不把缺失资格补成 0。
    """
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
    """功能：固定冷岗位划分；输入岗位 ID；输出岗位到 split 映射。
    逻辑：排序后使用独立 seed 置换；约束：要求恰好100个不同岗位，比例固定70/15/15。
    """
    ids = sorted(job_ids)
    if len(ids) != 100 or len(set(ids)) != 100:
        raise ValueError("Expected exactly 100 distinct jobs.")
    shuffled = np.random.default_rng(CONFIG["seed"]).permutation(ids)
    return {str(job): ("train" if i < 70 else "validation" if i < 85 else "test")
            for i, job in enumerate(shuffled)}


def parse_list(value: str) -> list[str]:
    """功能：解析源数据列表；输入 Python 列表文本；输出字符串列表。
    逻辑：literal_eval 后检查类型；约束：格式错误直接抛出，不隐式猜测分隔符。
    """
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise ValueError("Expected a list of strings in profile.")
    return parsed


def profile_texts(users: pd.DataFrame, jobs: pd.DataFrame) -> tuple[list[str], list[str]]:
    """功能：形成文本输入；输入完整档案表；输出候选文本/岗位查询文本。
    逻辑：技能/兴趣/教育/经验在前、合成描述在后；岗位使用固定资格检索指令。
    约束：不输入 ID、匹配标签、源 score 或虚构面试证据；后续按 token 上限截断。
    """
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
    """功能：计算11项显式匹配特征；输入配对和档案；输出 float32 矩阵。
    逻辑：技能覆盖、偏好/专业/模式匹配及数值余量；约束：资格标签可能由类似规则生成，
    因而本基线仅度量合成任务可学习性。没有人口属性和不存在的面试分。
    """
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
    """功能：完成实体映射与划分审计；输入源表/输出路径；输出全部训练准备对象。
    逻辑：合并配对、岗位隔离、验证每任务每划分均有正负例；约束：候选人允许跨划分，
    所以这是冷岗位而非冷候选人实验；候选池仅包含源数据明确标注的配对。
    """
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
    """功能：计算明确候选池的宏平均排名；输入已观察标签表/分数/task；输出指标字典。
    逻辑：pref 按候选人分组，qual 按岗位分组；无正例组不计均值并单独报告；同分按稳定
    输入顺序打破。约束：不是全库检索；nDCG/Recall 的分母只来自已标注候选池。
    """
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
    """功能：按任务和数据划分评测；输入配对表、N×2分数、split；输出两任务指标。
    逻辑：只选显式标签；约束：sigmoid/排序分数不被当作已校准业务概率。
    """
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
    """功能：计算 TF-IDF 配对余弦；输入配对/文本/目录；输出N×2共享语义分数。
    逻辑：只在训练配对涉及的实体文本上拟合词表，验证测试仅 transform；约束：此分数
    没有学习双边意愿。记录词表规模，不序列化含档案词汇的模型。
    """
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
    """功能：生成冻结 Qwen 表示；输入候选/岗位文本与目录；输出两侧归一化向量。
    逻辑：固定修订、左 padding、最后有效 token 池化；FP16/SDPA/cuda:0。
    约束：固定384 tokens，不因内存错误调整；仅保存向量和截断统计，不输出档案正文。
    """
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
    """功能：合并冻结向量和结构化特征；输入配对/向量/特征/目录；输出网络输入/余弦。
    逻辑：abs(u-v)、逐维乘积、训练侧标准化的结构化特征；约束：标准化不能使用验证测试。
    """
    u, j = embeddings[0][pairs.u_idx], embeddings[1][pairs.j_idx]
    scaler = StandardScaler().fit(structured[pairs.split.eq("train")])
    features = np.concatenate([np.abs(u - j), u * j, scaler.transform(structured)], axis=1).astype(np.float32)
    cosine = (u * j).sum(axis=1)
    write_json(output / "feature_scaler.json", {"names": STRUCTURED_NAMES,
               "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()})
    return features, np.column_stack([cosine, cosine])


class DualHeadModel(torch.nn.Module):
    """功能：偏好与资格共享表示网络；逻辑：MLP trunk 后分别线性输出。
    约束：两头独立标签掩码；实例状态 trunk/heads 只含训练权重，不含实体 ID。
    """

    def __init__(self, feature_size: int):
        """功能：初始化网络；输入特征维度；输出新实例。
        逻辑：固定128/32隐藏层和dropout；约束：网络参数与 CONFIG 一致。
        """
        super().__init__()
        first, second = CONFIG["hidden_sizes"]
        self.trunk = torch.nn.Sequential(torch.nn.Linear(feature_size, first), torch.nn.ReLU(),
                     torch.nn.Dropout(CONFIG["dropout"]), torch.nn.Linear(first, second), torch.nn.ReLU())
        self.heads = torch.nn.ModuleList([torch.nn.Linear(second, 1), torch.nn.Linear(second, 1)])

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """功能：输出偏好/资格 logits；输入 B×D；输出 B×2。
        逻辑：共享 trunk 后拼接两个线性头；约束：推理调用方负责 eval 和 sigmoid。
        """
        hidden = self.trunk(features)
        return torch.cat([head(hidden) for head in self.heads], dim=1)


def train_dual_head(pairs: pd.DataFrame, features: np.ndarray, output: Path) -> np.ndarray:
    """功能：训练双头模型；输入配对/特征/目录；输出所有配对两头分数。
    逻辑：缺失标签通过 mask 排除，训练侧正类权重，12轮中按验证双任务平均AP选最佳；
    约束：不查测试选择模型，不早停或自动调整参数；测试结果只在训练结束后计算。
    """
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
    """功能：结构化逻辑回归基线；输入配对/原始11维/目录；输出N×2分数。
    逻辑：各头只在各自训练标签上拟合标准化和逻辑回归；约束：无任务标签传播。
    """
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
    """功能：资格方向两阶段精排；输入配对/文本/余弦/目录；输出固定测试候选池的指标。
    逻辑：各岗位语义 top10，Qwen yes/no logits 重排前缀，尾部保持原次序。
    约束：固定 query/doc 各160/320token；不是全库召回；不报告伪概率AUC，不调整候选池。
    """
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
    """功能：输出可读报告；输入指标/目录；输出无，保存 Markdown。
    逻辑：分别展示任务及候选池指标；约束：不把合成标签结果写成真实招聘或面试增益。
    """
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
    """功能：执行完整实验；输入 Kaggle GPU/网络环境；输出 /kaggle/working/jobrec_v1。
    逻辑：各阶段及时保存审计，最终保存模型/预测/指标；约束：任何异常记录当前阶段并
    重新抛出，不自动重跑、降级或改参数。故障输出保留已完成阶段的证据。
    """
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
