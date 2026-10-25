"""Responsibilities: define the frozen v4 protocol, entity-level profile masking, and missing-aware features.
Implementation: derive deterministic masks from SHA256, hide source fields before feature calculation,
then compute 11 NaN-aware pairwise features.
Related Modules: jobrec_v1 supplies the established list parser and feature names; experiment trains;
jobrec_missing_eval evaluates.

Declaration Index:
- uniform: produce a deterministic uniform value from the registered identity tuple.
- scenarios: list the fixed evaluation scenarios and repeat counts.
- make_masks: create deterministic user/job field masks without using labels.
- hide_profiles: physically hide selected source values before feature construction.
- unknown: classify a scalar as missing without conflating invalid values with absence.
- number: parse known numeric values while preserving missingness.
- sequence: parse known list fields while preserving unknown versus empty.
- academic: map a supported academic-level code.
- boolean: convert a supported summer-commitment value.
- features: recompute the 11-dimensional pair feature matrix from visible source fields.
- feature_view: build the registered 11-, 22-, or 6-column model input.

Variable Index:
- USER_BLOCKS/JOB_BLOCKS: atomic user/job profile groups used for masking.
- OPTIONAL/CORE: fixed feature-column indices for the registered variants.
- LEVELS: supported academic-level encoding.
- RANKER_CONFIG: fixed historical tree parameters.
- PROTOCOL: approved v4 scenarios, model variants, and selection conditions.
- HISTORICAL: hashes for the v3 reference rankers and predictions.

Constraints: missingness is not chosen from labels; do not zero-fill or infer hidden values; do not
retry. IDs are used to derive masks but are not prediction features."""

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
    """Functionality: return a stable pseudo-random value in [0, 1).
    Inputs: namespace, scenario, repeat number, entity ID, and profile block.
    Outputs: deterministic float in [0, 1).
    Logic: serialize the fixed seed and identity tuple as compact JSON, hash it with SHA256, and map the
    first eight bytes to an unsigned integer fraction.
    Constraints: independent of row order and Python process hashing."""
    raw = json.dumps([42, namespace, scenario, repeat, str(entity), block], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") / 2**64


def scenarios() -> list[tuple[str, int]]:
    """Functionality: enumerate the fixed evaluation scenarios.
    Inputs: the module-level frozen protocol. Outputs: ordered scenario/repeat pairs.
    Logic: include each deterministic scenario once and five random scenarios five times each.
    Constraints: repeat zero is the primary report; repeats are not selected based on scores."""
    fixed = ["clean", "optional_bundle", "skills_only", "no_candidate_information"]
    fixed += ["single_" + str(i) for i in range(11)]
    random = ["random_10", "random_30", "random_50", "mixed_optional", "job_random_30"]
    return [(name, 0) for name in fixed] + [(name, r) for name in random for r in range(5)]


def make_masks(users: pd.DataFrame, jobs: pd.DataFrame, scenario: str, repeat: int = 0,
               namespace: str = "evaluation") -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Functionality: create user and job profile masks for a registered scenario.
    Inputs: entity tables with IDs, scenario, repeat index, and mask namespace.
    Outputs: user mask, job mask, and user assignment labels.
    Logic: select fixed blocks or hash-derived blocks; assign training-mix users to the 50/25/25 groups.
    Constraints: only IDs determine assignments; labels are never read."""
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
    """Functionality: create copies of profile tables with selected source fields hidden.
    Inputs: entity table, Boolean block mask, and block definitions.
    Outputs: an independent table with selected values set to None.
    Logic: cast affected columns to object before replacing selected cells.
    Constraints: never mutate the source table or retain hidden values for feature calculation."""
    if masks.shape != (len(frame), len(blocks)):
        raise ValueError("Mask shape differs from entity table")
    result = frame.copy()
    for col, block in enumerate(blocks):
        for name in block:
            result[name] = result[name].astype(object)
            result.loc[masks[:, col], name] = None
    return result


def unknown(value: object) -> bool:
    """Functionality: determine whether a scalar denotes unknown data.
    Inputs: one field value. Outputs: Boolean.
    Logic: only None and floating-point NaN are unknown.
    Constraints: empty strings, infinities, and invalid types are not converted to missing; list fields use
    sequence parsing."""
    return value is None or isinstance(value, (float, np.floating)) and np.isnan(value)


def number(value: object) -> float:
    """Functionality: parse one numeric profile field while preserving absence.
    Inputs: one scalar. Outputs: a finite float or NaN.
    Logic: retain values classified as unknown; convert supported known values to float.
    Constraints: reject infinity, Boolean values, and invalid text; preserve a known zero."""
    if unknown(value):
        return np.nan
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("Boolean is not a numeric quantity")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError("Invalid non-finite numeric quantity")
    return result


def sequence(value: object) -> set | None:
    """Functionality: parse a skill, interest, or major list while preserving unknown state.
    Inputs: one field value. Outputs: a set or None.
    Logic: use the established v1 parser for known values.
    Constraints: do not convert unknown to an empty set or alter case."""
    return None if unknown(value) else set(base.parse_list(value))


def academic(value: object) -> float:
    """Functionality: validate and map an academic-level code.
    Inputs: one code. Outputs: the supported original code or NaN.
    Logic: look up the fixed LEVELS mapping.
    Constraints: reject unsupported codes rather than treating them as ordinary missing data."""
    return np.nan if unknown(value) else float(LEVELS[value])


def boolean(value: object) -> float:
    """Functionality: convert the summer-commitment field.
    Inputs: one field value. Outputs: 0, 1, or NaN.
    Logic: accept only Boolean values or the registered true/false strings.
    Constraints: unknown does not mean false; invalid strings raise an error; valid source meaning is
    unchanged."""
    if unknown(value):
        return np.nan
    text = str(value).lower()
    if text not in ("true", "false"):
        raise ValueError("Invalid commitment boolean")
    return float(text == "true")


def features(pairs: pd.DataFrame, users: pd.DataFrame, jobs: pd.DataFrame) -> np.ndarray:
    """Functionality: recompute missing-aware pair features from masked source profiles.
    Inputs: pair-index table and the already-hidden user/job tables.
    Outputs: an N-by-11 float32 matrix.
    Logic: parse each entity once, compute set matches per pair, and vectorize numeric margins.
    Constraints: do not access a complete-profile cache. A partially known major set remains usable; only
    an entirely unknown major block yields NaN. Indices must refer to the supplied entity tables. An empty
    job-skill requirement preserves the historical error. NaN is allowed; infinity is rejected."""
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
    """Functionality: construct the input view for a registered model variant.
    Inputs: an N-by-11 feature array and variant name. Outputs: an 11-, 22-, or 6-column array.
    Logic: C appends NaN indicators, D selects the fixed six columns, and other variants copy the input.
    Constraints: reject unknown variants, infinity, and invalid dimensions."""
    if model not in PROTOCOL["models"] or matrix.ndim != 2 or matrix.shape[1] != 11 or np.isinf(matrix).any():
        raise ValueError("Invalid model or structured feature matrix")
    if model == "C":
        return np.concatenate([matrix, np.isnan(matrix).astype(np.float32)], axis=1)
    return matrix[:, CORE].copy() if model == "D" else matrix.copy()
