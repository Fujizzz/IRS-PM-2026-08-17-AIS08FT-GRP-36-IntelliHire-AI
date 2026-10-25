"""职责：定义v4冻结协议、实体级资料遮蔽与缺失感知特征。
实现：SHA256确定掩码；先隐藏原始字段，再计算NaN感知的11项配对特征。
关联：jobrec_v1提供既定列表解析和特征名；experiment负责训练，evaluation负责评测。
目录：uniform、scenarios、make_masks、hide_profiles、unknown、number、sequence、
      academic、boolean、features、feature_view。
变量索引：USER_BLOCKS/JOB_BLOCKS为资料块；OPTIONAL/CORE为列索引；LEVELS为学业编码；
      RANKER_CONFIG为历史树参数；PROTOCOL为已批准的新条件；HISTORICAL为v3产物哈希。
约束：不读取标签决定缺失；不补零、不推断隐藏字段、不自动重试；ID不是预测特征。
"""

import hashlib
import json

import numpy as np
import pandas as pd

import jobrec_v1 as base

USER_BLOCKS = [("skills",), ("interests",), ("masters_major", "second_major", "undergrad_major"),
               ("in_person_commitment",), ("gpa",), ("months_experience",), ("academic_level",),
               ("hours_per_week",), ("length_of_commitment",), ("num_publications",), ("commit_to_summer",)]
JOB_BLOCKS = [(x,) for x in ("required_skills", "industry", "acceptable_majors", "job_in_person_commitment",
              "min_gpa", "min_months_experience", "min_academic_level", "min_hours_per_week",
              "min_length_of_commitment", "min_num_publications")]
OPTIONAL = [4, 7, 8, 9, 10]
CORE = [0, 1, 2, 3, 5, 6]
LEVELS = {"UG1": 1, "UG2": 2, "UG3": 3, "UG4": 4, "MS1": 5, "MS2": 6,
          "PhD1": 7, "PhD2": 8, "PhD3": 9, "PhD4": 10, "PhD5": 11}
RANKER_CONFIG = {"objective": "lambdarank", "n_estimators": 200, "learning_rate": 0.05,
                 "num_leaves": 15, "min_child_samples": 20, "reg_lambda": 1.0,
                 "random_state": 42, "n_jobs": 2, "verbosity": -1,
                 "deterministic": True, "force_col_wise": True}
PROTOCOL = {"name": "jobrec-missing-v4", "seed": 42, "mask_replicates": 5,
            "training_mix": {"clean": 0.5, "random_30": 0.25, "optional_bundle": 0.25},
            "validation_scenarios": ["clean", "random_30", "optional_bundle"],
            "primary_scenarios": ["random_30", "optional_bundle", "mixed_optional"],
            "selection": "per-task equal mean validation AP; grid20..200; earliest tie",
            "models": ["H", "A", "B", "C", "D"], "tree_grid": list(range(20, 201, 20)),
            "bootstrap_replicates": 2000, "clean_max_drop": 0.02, "missing_min_gain": 0.03,
            "ranker": RANKER_CONFIG, "user_blocks": USER_BLOCKS, "job_blocks": JOB_BLOCKS,
            "optional_columns": OPTIONAL, "core_columns": CORE,
            "feature_names": list(base.STRUCTURED_NAMES), "lightgbm": "4.6.0",
            "scope": "development diagnostics; synthetic labels; no interview fusion or deployment"}
HISTORICAL = {
    "lambdarank/pref.txt": "7cf96ed020d367e9798893ffe7713674620073bda0941a7e1e8682758cb2d8cb",
    "lambdarank/qual.txt": "b0c310f4dcb66c375794fbe2f931a19ca108ab17709d64ef0fc6d0daca9b00e7",
    "test_predictions.csv": "1eff7faac87ec5011f70c247898935d7569169d3dcc28337fc3d1aa5d4431f62",
    "metrics.json": "577b26e0b871adbbf7f3e98806b5e862a805e3d95178c437be28ff6b7a676a2d",
}


def uniform(namespace: str, scenario: str, repeat: int, entity: str, block: str) -> float:
    """功能：生成稳定均匀值；输入用途/场景/重复/实体/块；输出[0,1)数。
    逻辑：seed42和参数JSON的SHA256前8字节转无符号数；约束：不依赖行序或进程hash。
    """
    raw = json.dumps([42, namespace, scenario, repeat, str(entity), block], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") / 2**64


def scenarios() -> list[tuple[str, int]]:
    """功能：列举固定评测清单；输入模块协议；输出场景/重复序列。
    逻辑：确定性场景一次，五种随机场景各五次；约束：r0主报告，不按成绩选择重复。
    """
    fixed = ["clean", "optional_bundle", "skills_only", "no_candidate_information"]
    fixed += ["single_" + str(i) for i in range(11)]
    random = ["random_10", "random_30", "random_50", "mixed_optional", "job_random_30"]
    return [(name, 0) for name in fixed] + [(name, r) for name in random for r in range(5)]


def make_masks(users: pd.DataFrame, jobs: pd.DataFrame, scenario: str, repeat: int = 0,
               namespace: str = "evaluation") -> tuple[np.ndarray, np.ndarray, list[str]]:
    """功能：生成实体资料块缺失掩码；输入含ID的实体表/场景/重复/用途；输出两掩码及用户分配。
    逻辑：固定或SHA抽样选择块，training_mix先分配50/25/25类别；约束：只读ID，不读标签。
    """
    allowed = {name for name, _ in scenarios()} | {"training_mix"}
    if scenario not in allowed or repeat not in range(5):
        raise ValueError("Unregistered mask scenario or repeat")
    um = np.zeros((len(users), 11), dtype=bool)
    jm = np.zeros((len(jobs), 10), dtype=bool)
    assignments = []
    for row, entity in enumerate(users.student_id):
        mode = scenario
        if scenario == "training_mix":
            choice = uniform(namespace, scenario, repeat, entity, "assignment")
            mode = "clean" if choice < .5 else "random_30" if choice < .75 else "optional_bundle"
        assignments.append(mode)
        if mode.startswith("single_"):
            um[row, int(mode.split("_")[1])] = True
        elif mode == "optional_bundle":
            um[row, OPTIONAL] = True
        elif mode == "mixed_optional":
            if uniform(namespace, scenario, repeat, entity, "assignment") < .5:
                um[row, OPTIONAL] = True
        elif mode == "skills_only":
            um[row, 1:] = True
        elif mode == "no_candidate_information":
            um[row] = True
        elif mode.startswith("random_"):
            probability = int(mode.split("_")[1]) / 100
            um[row] = [uniform(namespace, scenario, repeat, entity, "/".join(block)) < probability
                       for block in USER_BLOCKS]
    if scenario == "job_random_30":
        for row, entity in enumerate(jobs.job_id):
            jm[row] = [uniform(namespace, scenario, repeat, entity, "/".join(block)) < .3
                       for block in JOB_BLOCKS]
    return um, jm, assignments


def hide_profiles(frame: pd.DataFrame, masks: np.ndarray, blocks: list) -> pd.DataFrame:
    """功能：物理隐藏原始字段；输入实体表/块掩码/块定义；输出独立资料表。
    逻辑：相关列转object后将选择值置None；约束：不修改源表，不保留隐藏值供特征计算。
    """
    if masks.shape != (len(frame), len(blocks)):
        raise ValueError("Mask shape differs from entity table")
    result = frame.copy()
    for col, block in enumerate(blocks):
        for name in block:
            result[name] = result[name].astype(object)
            result.loc[masks[:, col], name] = None
    return result


def unknown(value: object) -> bool:
    """功能：判定标量未知；输入单字段值；输出布尔；逻辑：仅None或浮点NaN为未知。
    约束：空字符串、无穷大、错误类型不伪装成缺失，列表由sequence单独解析。
    """
    return value is None or isinstance(value, (float, np.floating)) and np.isnan(value)


def number(value: object) -> float:
    """功能：解析已知数值；输入标量；输出float或NaN；逻辑：未知保留，其他转float。
    约束：无穷大、布尔和非法文本报错，已知零不改写。
    """
    if unknown(value):
        return np.nan
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("Boolean is not a numeric quantity")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError("Invalid non-finite numeric quantity")
    return result


def sequence(value: object) -> set | None:
    """功能：解析技能/兴趣/专业列表；输入字段；输出集合或None。
    逻辑：已知字段沿用v1解析器；约束：未知不会变为空集合，不调整大小写。
    """
    return None if unknown(value) else set(base.parse_list(value))


def academic(value: object) -> float:
    """功能：学业阶段映射；输入代码；输出原编码或NaN；逻辑：查固定LEVELS。
    约束：未知代码报错，不将未支持代码视为普通缺失。
    """
    return np.nan if unknown(value) else float(LEVELS[value])


def boolean(value: object) -> float:
    """功能：暑期布尔转换；输入值；输出0/1/NaN；逻辑：只接受true/false字符串或布尔。
    约束：未知不映射false，错误字符串报错；原有效数据的含义保持不变。
    """
    if unknown(value):
        return np.nan
    text = str(value).lower()
    if text not in ("true", "false"):
        raise ValueError("Invalid commitment boolean")
    return float(text == "true")


def features(pairs: pd.DataFrame, users: pd.DataFrame, jobs: pd.DataFrame) -> np.ndarray:
    """功能：重算缺失感知匹配；输入配对索引和已隐藏原始资料；输出N×11 float32。
    逻辑：解析实体一次，集合匹配逐配对，数值差按索引向量化；约束：不访问完整资料缓存。
    专业以已知专业集合为依据，原无第二/硕士专业不使本科信息失效；整个专业块未知才NaN。
    输入u_idx/j_idx必须对应给定实体表；空岗位技能沿用历史报错；NaN允许，无穷大拒绝。
    """
    ui, ji = pairs.u_idx.to_numpy(), pairs.j_idx.to_numpy()
    us = [sequence(v) for v in users.skills]
    req = [sequence(v) for v in jobs.required_skills]
    if any(v is not None and not v for v in req):
        raise ValueError("Empty required skill list is outside the frozen protocol")
    interests = [sequence(v) for v in users.interests]
    majors = [{v for v in row if not unknown(v)} or None
              for row in users[["masters_major", "second_major", "undergrad_major"]].itertuples(index=False, name=None)]
    accepted = [sequence(v) for v in jobs.acceptable_majors]
    industries = jobs.industry.tolist()
    modes_u, modes_j = users.in_person_commitment.tolist(), jobs.job_in_person_commitment.tolist()
    result = np.full((len(pairs), 11), np.nan, dtype=np.float32)
    for k, (u, j) in enumerate(zip(ui, ji)):
        if us[u] is not None and req[j] is not None:
            result[k, 0] = len(us[u] & req[j]) / len(req[j])
        if interests[u] is not None and not unknown(industries[j]):
            result[k, 1] = float(industries[j] in interests[u])
        if majors[u] is not None and accepted[j] is not None:
            result[k, 2] = float(bool(majors[u] & accepted[j]))
        if not unknown(modes_u[u]) and not unknown(modes_j[j]):
            result[k, 3] = float(modes_u[u] == modes_j[j])
    for c in range(4, 10):
        parser = academic if c == 6 else number
        uv = np.asarray([parser(v) for v in users[USER_BLOCKS[c][0]]])
        jv = np.asarray([parser(v) for v in jobs[JOB_BLOCKS[c][0]]])
        result[:, c] = uv[ui] - jv[ji]
    result[:, 10] = np.asarray([boolean(v) for v in users.commit_to_summer])[ui]
    if np.isinf(result).any():
        raise ValueError("Infinite feature")
    return result


def feature_view(matrix: np.ndarray, model: str) -> np.ndarray:
    """功能：提供模型输入视图；输入N×11数组/模型名；输出11/22/6维数组。
    逻辑：C添加NaN指示，D取固定六列，其余复制；约束：拒绝未知模型、无穷大和维度错。
    """
    if model not in PROTOCOL["models"] or matrix.ndim != 2 or matrix.shape[1] != 11 or np.isinf(matrix).any():
        raise ValueError("Invalid model or structured feature matrix")
    if model == "C":
        return np.concatenate([matrix, np.isnan(matrix).astype(np.float32)], axis=1)
    return matrix[:, CORE].copy() if model == "D" else matrix.copy()
