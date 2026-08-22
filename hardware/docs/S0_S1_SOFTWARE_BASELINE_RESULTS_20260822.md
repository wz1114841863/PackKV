# BRISK-KV S0/S1 Software Baseline Results

Date: 2026-08-22

Scope: frozen-hardware paper preparation. This record reports software
algorithm/format evidence only. It does not modify the compression algorithm,
RTL, or frozen hardware artifacts.

## 1. Completion Status

| Stage | Status | Evidence |
|---|---|---|
| S0: frozen-boundary and artifact preparation | Complete | `hardware/docs/SOFTWARE_BASELINE_S0_S1_PREP_20260821.md`; all new results are isolated under `paper_s1_20260821_*`. |
| S1: unified software-baseline reconstruction | Complete | `paper_s1_20260821_full`, check report status `pass`, 12 accuracy rows and 9 CR rows. |

S1 completed the planned fixed matrix:

- Models: `Qwen/Qwen3-4B`, `Qwen/Qwen3-8B`, and
  `NousResearch/Meta-Llama-3.1-8B`.
- Accuracy task: full GSM8K, batch size 1.
- CR context length: 4096 tokens.
- Variants: FP, continuous PackKV, PO2+NONE, and PO2+BUCKET.
- Common PackKV parameters: K scale 0.03, V scale 0.10, block 64, buffer 192,
  pack 16, four `k_sum` buckets, and integer zero point.

The pilot run (`paper_s1_20260821_pilot`) is a 20-example environment and
schema smoke test only. It is not a paper data point.

## 2. Full-Matrix Results

All accuracy figures are fractions from the source CSV, displayed below as
percentages. `CR=1.0` for FP is a reference label rather than a compressed CR
measurement.

| Model | Variant | Strict | Flexible | Overall CR |
|---|---|---:|---:|---:|
| Qwen3-4B | FP | 85.2919% | 84.9886% | 1.0000x |
| Qwen3-4B | continuous | 69.9014% | 80.7430% | 3.9801x |
| Qwen3-4B | PO2+NONE | 71.1903% | 80.1365% | 3.5409x |
| Qwen3-4B | PO2+BUCKET | 71.2661% | 80.2881% | 3.7898x |
| Qwen3-8B | FP | 87.6422% | 88.0970% | 1.0000x |
| Qwen3-8B | continuous | 85.8984% | 76.1941% | 3.9827x |
| Qwen3-8B | PO2+NONE | 87.1114% | 84.4579% | 3.5295x |
| Qwen3-8B | PO2+BUCKET | 87.1873% | 84.6096% | 3.7772x |
| Llama-3.1-8B | FP | 50.1137% | 50.0379% | 1.0000x |
| Llama-3.1-8B | continuous | 49.5072% | 49.7346% | 4.0751x |
| Llama-3.1-8B | PO2+NONE | 50.3412% | 50.2654% | 3.6223x |
| Llama-3.1-8B | PO2+BUCKET | 50.2654% | 50.1895% | 3.8423x |

For the retained PO2 point, BUCKET improves overall CR over NONE by 7.0293%
(Qwen3-4B), 7.0180% (Qwen3-8B), and 6.0735% (Llama-3.1-8B).

The NONE/BUCKET paired accuracy audit covers 2,638 sample/filter keys per
model with no missing keys. BUCKET minus NONE strict deltas are +0.0758 pp,
+0.0758 pp, and -0.0758 pp for Qwen3-4B, Qwen3-8B, and Llama-3.1-8B
respectively. These are one-label-scale differences and must not be described
as an accuracy improvement or degradation.

## 3. Evidence and Integrity Checks

Primary full-run summaries:

- `csv_results/paper_s1_20260821_full/accuracy_summary.csv`
- `csv_results/paper_s1_20260821_full/repack_accuracy_comparison.csv`
- `csv_results/paper_s1_20260821_full/joint_accuracy_cr_summary.csv`
- `csv_results/paper_s1_20260821_full/s1_check_report.json`

Raw accuracy evidence:

- `eval_logs/paper_s1_20260821_full/` contains 12 `results.json`, 12
  `packkv_config.json`, and 12 `samples_gsm8k.json` files.

CR evidence:

- `csv_results/Global_Macro_Summary_v11-2.csv` contains the nine full rows
  selected by `Suite_ID=paper_s1_20260821_full-cr`.
- Nine corresponding `csv_results/CR_*.csv` reports are present. Their
  model/configuration fields and layer counts match the macro rows exactly.
- `csv_results/Layer_Detail_v6.csv` contains 312 non-duplicate full-run layer
  rows: 36 layers for each Qwen configuration and 32 layers for each Llama
  configuration.
- Every full macro row has zero K/V accounting error. Recomputing overall CR
  from original and compressed byte columns matches the reported value.

SHA-256 at record creation:

| Artifact | SHA-256 |
|---|---|
| `csv_results/Global_Macro_Summary_v11-2.csv` | `498e3dfb572184f0c0fca7ec8b94f4b50b2e0ac35c3e70676dea8671d8c0540e` |
| `csv_results/Layer_Detail_v6.csv` | `b58f2bc06efebe0b19dc5ecfa86eea4a1e0e2c65775c88e666f8cce567917ba5` |
| full `accuracy_summary.csv` | `045c19620b346ce68bb5415b83e02fb80f081d1db4c100faf016d07a66de2609` |
| full `repack_accuracy_comparison.csv` | `242647009219c3e5f069f6b5ab42d5c93f65c92b41c7a67878758cf617c7860b` |
| full `joint_accuracy_cr_summary.csv` | `48e4d4b029209e2aaa443e9131b58270994bddff3e3dee5a5f5cb59d4a0fcd8d` |
| full `s1_check_report.json` | `c5e211f64669de30502bdf5a5f93ff8424fce410333ecfcaa99ca1b124228d9f` |

## 4. Allowed Paper Use and Limitations

This evidence is sufficient for the paper's S1 algorithm/format table:
FP -> continuous PackKV -> PO2+NONE -> PO2+BUCKET, under the stated GSM8K and
4096-token CR conditions. It supports the limited statement that stable
four-bucket repacking improves the retained PO2 point's measured CR by about
6%-7% versus NONE, with small paired NONE/BUCKET accuracy movement.

Do not claim that compression improves accuracy. FP-relative effects must be
reported exactly as shown above. In particular, continuous and PO2 variants
have model- and metric-dependent accuracy movement.

This stage does not support GPU speed/energy claims, full-request power,
context-length scaling, multi-head or full-system claims, or real-model
software-to-RTL output equivalence. Those belong to later planned stages.

Reproducibility caveat: `results.json` records fixed random seeds, but the
model metadata currently uses `model_revision=main` and has an empty
`model_sha`; a final submission archive should additionally freeze actual model
commits and the GSM8K dataset revision/fingerprint.
