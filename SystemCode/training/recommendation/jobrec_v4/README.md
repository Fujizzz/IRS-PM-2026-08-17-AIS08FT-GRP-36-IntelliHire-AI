# JobRec v4 Kaggle training package

This directory keeps the source snapshot used by the Kaggle kernel `heropig/irs-jobrec-missing-v4`, version 1. It is the training experiment corresponding to the `jobrec-v4-B` LightGBM bundle currently loaded by the backend. The MiniLM answer-completion training pipeline is maintained separately under `training/answer_completion/`.

## Contents

| File | Responsibility |
| --- | --- |
| `jobrec_v1.py` | Frozen data preparation, feature, metric, and environment helpers imported by later experiment modules. |
| `jobrec_v2.py` | Historical input verification and paired-query bootstrap helpers. |
| `jobrec_missing.py` | v4 protocol, deterministic missing-profile masks, feature reconstruction, and model parameters. |
| `jobrec_missing_eval.py` | Submitted v4 evaluation of fixed predictions, query-level metrics, bootstrap intervals, and candidate gates. |
| `experiment.py` | Reproduces the clean historical baseline, fits A/B/C/D model pairs, and records predictions and evaluation inputs. |
| `build_notebook.py` | Verifies the submitted source hashes and creates the Kaggle notebook plus private kernel metadata. |
| `protocol.json` | Machine-readable v4 training and evaluation conditions. |
| `source_hashes.json` | Provenance and SHA256 identities extracted from the Kaggle-pulled submitted notebook. |

`jobrec_missing_eval.py` is the exact evaluator embedded in the Kaggle submission. The separate research workspace later corrected a reporting defect in its local post-run analysis; that correction did not change model fitting, labels, feature construction, or predictions. Keeping the submitted evaluator here preserves the source identity of the training run.

## Rebuild the Kaggle submission

Run from the `SystemCode/` directory:

```powershell
python training/recommendation/jobrec_v4/build_notebook.py
kaggle kernels push -p training/recommendation/jobrec_v4/submission
```

The kernel is private, internet-enabled, and requests the original T4 environment. Its inputs must include `heropig/irs-jobrec-feasibility-v1` and `heropig/irs-jobrec-controls-v3`; the experiment verifies the expected parent artifacts before fitting. The tree models themselves train on CPU with two threads. Building the notebook is local; the `kaggle kernels push` command submits it to Kaggle.

## Result status and limits

The recorded Kaggle run completed eight new fits (A/B/C/D, each with preference and qualification rankers). The backend package exposes `jobrec-v4-B` as an experimental ranking model. Its manifest records `release_gate_passed: false`, and the v4 report did not meet the pre-registered candidate gate. `qual` is a synthetic per-job winner-ranking target; it is not a qualification probability, hiring probability, or real employer preference model. The run is not evidence that the model passed release validation.

The model outputs, source datasets, Kaggle logs, and local LightGBM installation are not copied into this source package. The backend inference bundle remains at `backend/interviews/recommendation/artifacts/jobrec-v4-B/`.
