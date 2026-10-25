"""Responsibilities: generate reproducible synthetic bilingual data for semantic answer-completion detection.

Implementation: preassign source groups to splits, generate contrastive variants of fictional interview
answers with Qwen Plus, validate structure and exact deduplication, then save provenance.
Related Modules: experiment.json fixes conditions before training; train.py consumes JSONL only.

Declaration Index:
- generate_group: generate one shared-context semantic family without assigning labels from cue words.
- parse_group: validate saved response categories and attach unchanged group/label provenance.
- apply_corrections: apply explicit checksum-bound semantic rewrites without changing labels or splits.
- write_jsonl: persist UTF-8 records without credentials or provider response envelopes.
- main: resolve explicit output/config, generate bounded concurrent groups, and save dataset metadata.

Variable Index:
- CATEGORIES: semantic generation conditions and target labels; never runtime cue rules.
- TOPICS: fictional interview topic pool shared across languages.
- SYSTEM_PROMPT: generation contract requiring same-context positives and hard negatives.

Constraints:
Labels are scenario-conditioned synthetic labels, not independent human ground truth. A failed
request or invalid response aborts without retries. Split and random seed are fixed before generation;
source groups never cross splits. Real candidate data, public tests, and secrets are not uploaded."""

import argparse
import asyncio
import hashlib
import json
import os
import random
import unicodedata
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

CATEGORIES = {
    "direct_end": 1,
    "indirect_end": 1,
    "yield_next": 1,
    "nothing_more": 1,
    "ordinary_content": 0,
    "task_complete": 0,
    "quoted_end": 0,
    "negated_end": 0,
    "continue_after_end": 0,
    "polite_only": 0,
}
TOPICS = (
    "database transactions", "cache consistency", "team conflict", "project migration",
    "incident response", "test design", "cloud deployment", "data cleaning",
    "model evaluation", "API security", "batch processing", "leadership experience",
    "distributed locks", "monitoring", "requirements negotiation", "memory management",
    "accessibility", "queue design", "UI state transitions", "deadline pressure",
)
SYSTEM_PROMPT = (
    "Generate fictional interview candidate transcript tails for semantic end-intent training. "
    "Return JSON only: {\"samples\":[{\"category\":...,\"text\":...},...]}. "
    "Exactly one sample per requested category, ten total. Each text is 30-240 characters, "
    "a natural first-person spoken transcript (no speaker tags, markup, or category labels). "
    "Share an underlying answer scenario across all ten variants. Vary the wording creatively. "
    "Categories: direct_end explicitly ends this answer; indirect_end semantically communicates "
    "the candidate has exhausted what they want to say without a canned end phrase; yield_next "
    "ends this answer and yields to the next interview question; nothing_more explicitly has no "
    "further content to contribute to this answer. These four END THEIR OWN ANSWER NOW. "
    "ordinary_content describes technical work/results only, without end intent; task_complete "
    "describes a finished project/build/action, not finished speaking; quoted_end quotes an end "
    "phrase in a UI label, example, or another person's speech WITHOUT ending this answer; "
    "negated_end explicitly is NOT done answering; continue_after_end retracts an ending and "
    "continues with an actual detail; polite_only thanks the interviewer plus ordinary content, "
    "without closing the answer. These six DO NOT END THE ANSWER. "
    "Reuse completion-related words between positives and negatives so word occurrence alone "
    "cannot determine the label. Both classes must have comparable lengths and answer details. "
    "Every sample MUST begin with at least one concrete answer detail before its intent wording; "
    "do not output a short isolated closing phrase. For Chinese include at least 20 Chinese "
    "characters of answer details in each sample; for English at least one full detail sentence. "
    "Use varied colloquial paraphrases, contractions, and natural ASR punctuation variation. "
    "Do not use a single stock template. Do not mention AI or training."
)


async def generate_group(client, config, group, semaphore, output, profile):
    """Inputs: no-retry client, config/group, semaphore, output directory and explicit data profile.
    Outputs: ten provenance-bearing records; malformed categories abort generation. Length and
    duplicate validation occur during final assembly after any explicit reviewed corrections.
    Logic: semantic scenario determines labels; no search for completion words is used to label.
    Constraints: implicit teacher-generated labels require separate human/real-world validation.
    """
    async with semaphore:
        response = await client.chat.completions.create(
            model=config["generation_model"],
            messages=[
                {"role": "system", "content": profile["system_prompt"]},
                {"role": "user", "content": json.dumps(group, ensure_ascii=False)},
            ],
            temperature=config["generation_temperature"],
            max_tokens=4096,
            response_format={"type": "json_object"},
            extra_body={"enable_thinking": False},
        )
    choice = response.choices[0]
    (output / (group["group_id"] + ".json")).write_text(
        choice.message.content or "null", encoding="utf-8"
    )
    if choice.finish_reason != "stop":
        raise ValueError(f"Generation truncated for {group['group_id']}")
    return parse_group(choice.message.content, group, profile)


def parse_group(content, group, profile):
    """Inputs: response text, group and selected categories. Outputs: provenance-bearing records.
    Logic: validate category coverage; defer length/dedup validation to final assembly so all raw
    groups remain reviewable before explicit corrections. Label by scenario only.
    Constraints: malformed structure aborts; neither length limits nor source rows are changed.
    """
    samples = json.loads(content)["samples"]
    categories = profile["categories"]
    if len(samples) != len(categories) or {x["category"] for x in samples} != set(categories):
        raise ValueError(f"Generation categories invalid for {group['group_id']}")
    records = []
    for item in samples:
        text = item["text"].strip()
        records.append({
            **group, "id": group["group_id"] + "/" + item["category"],
            "text": text, "label": categories[item["category"]], "category": item["category"],
            "label_source": "scenario_conditioned_qwen_plus_synthetic",
        })
    print("generated", group["group_id"], "split", group["split"], flush=True)
    return records


def apply_corrections(records, corrections):
    """Inputs: generated records and explicit ID/old-hash/new-text correction entries.
    Outputs: corrected records in place; labels, source groups and splits remain unchanged.
    Logic: verify original bytes before assistant-authored semantic-equivalent rewrites; attach
    provenance. Constraints: no implicit repairs, dropped samples, or correction of model outcomes.
    """
    by_id = {item["id"]: item for item in records}
    for correction in corrections:
        item = by_id[correction["id"]]
        old_hash = hashlib.sha256(item["text"].encode()).hexdigest()
        if old_hash != correction["original_sha256"]:
            raise ValueError("Data correction does not match original generated record")
        text = correction["replacement_text"].strip()
        if not 15 <= len(text) <= 400:
            raise ValueError("Corrected text violates the original length contract")
        item["text"] = text
        item["correction_source"] = "assistant_reviewed_semantic_equivalent_rewrite"
        item["original_text_sha256"] = old_hash
        print("explicit_semantic_rewrite", item["id"], flush=True)


def write_jsonl(path, records):
    """Inputs: destination and JSON-serializable records. Outputs: UTF-8 JSONL file.
    Logic: preserve actual text and provenance; failure propagates, no training occurs here.
    """
    path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records), encoding="utf-8"
    )


async def main():
    """Inputs: output/config/profile, explicit checkpoint continuation/corrections and API env.
    Outputs: JSONL/provenance.
    Logic: deterministic per-language source split, bounded generation, Unicode exact dedup check.
    Constraints: output must not contain a dataset already; API errors are terminal and logged by
    exception location, not silently retried; --resume-checkpoints explicitly preserves completed
    responses and generates only missing groups. Secrets never enter saved metadata.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("experiment.json"))
    parser.add_argument("--checkpoints-only", action="store_true")
    parser.add_argument("--resume-checkpoints", action="store_true")
    parser.add_argument("--corrections", type=Path)
    parser.add_argument("--profile", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoints = args.output / "groups"
    checkpoints.mkdir(exist_ok=True)
    if (args.output / "data.jsonl").exists():
        raise FileExistsError("Refusing to regenerate an existing experiment dataset")
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    profile = (
        json.loads(args.profile.read_text(encoding="utf-8")) if args.profile is not None
        else {"categories": CATEGORIES, "system_prompt": SYSTEM_PROMPT}
    )
    corrections = (
        json.loads(args.corrections.read_text(encoding="utf-8")) if args.corrections else []
    )
    groups = []
    rng = random.Random(config["seed"])
    for language in ("zh", "en"):
        order = list(range(config["groups_per_language"]))
        rng.shuffle(order)
        for position, index in enumerate(order):
            train_end = config["train_groups_per_language"]
            validation_end = train_end + config["validation_groups_per_language"]
            split = "train" if position < train_end else (
                "validation" if position < validation_end else "test"
            )
            groups.append({
                "group_id": f"{language}-{index:03}", "language": language, "split": split,
                "topic": TOPICS[index % len(TOPICS)], "scenario_variation": index,
            })
    if args.checkpoints_only:
        nested = [parse_group((checkpoints / (g["group_id"] + ".json")).read_text(
            encoding="utf-8"
        ), g, profile) for g in groups]
    else:
        client = AsyncOpenAI(
            api_key=os.environ["DASHSCOPE_API_KEY"],
            base_url=os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            timeout=config["generation_timeout_seconds"], max_retries=0,
        )
        semaphore = asyncio.Semaphore(config["generation_concurrency"])
        try:
            completed = []
            missing = []
            for group in groups:
                saved = checkpoints / (group["group_id"] + ".json")
                if args.resume_checkpoints and saved.exists():
                    completed.append(parse_group(saved.read_text(encoding="utf-8"), group,
                                                 profile))
                else:
                    missing.append(group)
            tasks = [asyncio.create_task(generate_group(
                client, config, g, semaphore, checkpoints, profile
            )) for g in missing]
            try:
                nested = completed + await asyncio.gather(*tasks)
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            await client.close()
    records = [item for group in nested for item in group]
    if {c["id"] for c in corrections} - {r["id"] for r in records}:
        raise ValueError("Explicit correction refers to an unknown dataset record")
    apply_corrections(records, corrections)
    for item in records:
        if not 15 <= len(item["text"]) <= 400:
            raise ValueError(f"Generation length invalid for {item['id']}: {len(item['text'])}")
    normalized = [unicodedata.normalize("NFKC", r["text"]).casefold() for r in records]
    if len(set(normalized)) != len(normalized):
        raise ValueError("Duplicate texts found; preserve experiment conditions and review")
    write_jsonl(args.output / "data.jsonl", records)
    (args.output / "experiment.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "provenance.json").write_text(json.dumps({
        "records": len(records), "groups": len(groups), "public_training_data": "None",
        "human_label_validation": "None; generation and assistant qualitative review only",
        "explicit_data_corrections": corrections,
        "prompt_sha256": hashlib.sha256(profile["system_prompt"].encode()).hexdigest(),
        "generation_profile": profile,
        "data_sha256": hashlib.sha256((args.output / "data.jsonl").read_bytes()).hexdigest(),
        "config": config,
    }, indent=2), encoding="utf-8")
    print("saved_records", len(records), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
