"""Responsibilities: package the verified JobRec v4 source modules for Kaggle execution.
Implementation: verify every source hash, register modules in dependency order, then append the
v4 training entry point and write a private T4 kernel manifest.
Related Modules: the five adjacent Python sources define the frozen v1/v2 helpers and v4 protocol.

Declaration Index:
- module_cell: wrap a source file as an isolated notebook module.
- main: verify source identity and generate the Kaggle notebook and metadata.

Variable Index:
- SOURCES: ordered module filenames and immutable SHA256 values from the submitted Kaggle run.
- EXPERIMENT_SHA256: immutable SHA256 for the v4 training entry point.
- KERNEL_METADATA: private Kaggle kernel identity, resources, and parent kernel inputs.
"""

import hashlib
import json
from pathlib import Path

SOURCES = [
    ("jobrec_v1", "jobrec_v1.py", "300883ef6e12e1b7ac3697eedecdb20c9e9e121113d9d461477bf2b4e792305e"),
    ("jobrec_v2", "jobrec_v2.py", "d27b7bdb985468bf61d8ff84ce0b2fb888d73ab9d239fcc5b5dc8dce7c576991"),
    ("jobrec_missing", "jobrec_missing.py", "227b82a6db8b7072fc2938024d1282fe7af7953659e9e375b9460fc88c4967d3"),
    ("jobrec_missing_eval", "jobrec_missing_eval.py", "c0bd8081ca731b832e1315640dbaa4ea64a5eeb6f52b02a344c1a98714f8f02b"),
]
EXPERIMENT_SHA256 = "be356b627c96c759b7b075281dd129ca95d0f82e0712e7f8df6661cd86d9d344"
KERNEL_METADATA = {
    "id": "heropig/irs-jobrec-missing-v4",
    "title": "IRS JobRec Missing V4",
    "code_file": "experiment.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_internet": True,
    "machine_shape": "NvidiaTeslaT4",
    "kernel_sources": [
        "heropig/irs-jobrec-feasibility-v1",
        "heropig/irs-jobrec-controls-v3",
    ],
    "dataset_sources": [],
    "competition_sources": [],
    "model_sources": [],
}


def module_cell(name: str, source: str) -> str:
    """Functionality: create a notebook cell that registers one source module.

    Inputs: a unique module name and its complete Python source. Outputs: executable cell text.
    Logic: compile the source into a ModuleType namespace and register it in sys.modules so later
    modules can import its declared name without invoking its main guard.
    Constraints: source text is hash-checked by main before packaging; this function makes no file,
    network, Kaggle, model-training, or runtime configuration changes.
    """
    return (
        '"""Responsibilities: register a frozen experiment module.\n'
        'Implementation: compile source in an isolated module namespace.\n'
        'Related Modules: later notebook cells import this registered module.\n'
        'Declaration Index: None.\n'
        'Variable Index: source is embedded Python; module is the registered namespace.\n'
        '"""\n'
        "import sys\nimport types\n"
        f"source = {source!r}\n"
        f"module = types.ModuleType({name!r})\n"
        f"exec(compile(source, {name + '.py'!r}, 'exec'), module.__dict__)\n"
        f"sys.modules[{name!r}] = module\n"
    )


def main() -> None:
    """Functionality: build the versioned Kaggle notebook and kernel metadata.

    Inputs: adjacent source files and their fixed hashes. Outputs: submission/experiment.ipynb and
    submission/kernel-metadata.json; prints a success message after both writes complete.
    Logic: verify all five submitted source hashes, preserve dependency order, and serialize a
    notebook with empty outputs so execution begins only on Kaggle.
    Constraints: hash mismatches raise ValueError before writing; no Kaggle command or training is
    run, and the kernel remains private with its original parent-kernel dependencies.
    """
    folder = Path(__file__).resolve().parent
    cells, hashes = [], {}
    for name, filename, expected in SOURCES:
        source = (folder / filename).read_text(encoding="utf-8")
        digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
        if digest != expected:
            raise ValueError(f"Submitted source changed: {filename}")
        hashes[filename] = digest
        cells.append(module_cell(name, source))

    program = (folder / "experiment.py").read_text(encoding="utf-8")
    digest = hashlib.sha256(program.encode("utf-8")).hexdigest()
    if digest != EXPERIMENT_SHA256:
        raise ValueError("Submitted source changed: experiment.py")
    hashes["experiment.py"] = digest
    cells.append(program)

    notebook = {
        "nbformat": 4,
        "nbformat_minor": 4,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            }
        },
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": code.splitlines(keepends=True),
            }
            for code in cells
        ],
    }
    destination = folder / "submission"
    destination.mkdir(exist_ok=True)
    (destination / "experiment.ipynb").write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    (destination / "kernel-metadata.json").write_text(
        json.dumps(KERNEL_METADATA, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Built private JobRec v4 notebook; verified {len(hashes)} submitted source files")


if __name__ == "__main__":
    main()
