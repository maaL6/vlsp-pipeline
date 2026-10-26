# Claim-level and local-evidence audit (2026-10-27)

## Literature claims

This checks the passages relevant to the claims retained in the paper, beyond bibliographic metadata. It is not a certificate for every statement or benchmark in the cited papers. No cited paper is used as evidence for our own experimental results.

| Citation key | Claim in this manuscript | Passage checked | Source | Assessment |
|---|---|---|---|---|
| `vlsp2026` | The task constrains models below 3B, local/offline inference, and recommends public-test system selection. | Official task: Technical constraints; Evaluation; Development and Test Data. | [Primary source](https://vlsp.org.vn/vlsp2026/eval/mt_challenge) | Supported; do not present local scores as official private-test ranking. |
| `hu2022lora` | LoRA freezes W0 and learns BA scaled by alpha/r. | Section 4.1, low-rank parametrised update matrices. | [Primary source](https://arxiv.org/html/2106.09685v2#S4.SS1) | Supported; local dropout/target modules are implementation settings, not supplied by this citation. |
| `xu2024alma` | ALMA uses monolingual fine-tuning followed by high-quality parallel-data fine-tuning. | Section 4, A New Training Recipe. | [Primary source](https://arxiv.org/html/2309.11674v2#S4) | Supported; no claim that ALMA evaluated the current legal task. |
| `alves2024tower` | Tower combines continued pre-training and instruction tuning. | Sections 2.1 and 2.3; Section 4 training-recipe ablations. | [Primary source](https://arxiv.org/html/2402.17733v1) | Supported; Tower is context for model adaptation, not a measured baseline in this paper. |
| `zheng2026hymt2` | Hy-MT2 includes a 1.8B variant and supports 33 languages. | Abstract and Section 1, From Hy-MT1.5 to Hy-MT2. | [Primary source](https://arxiv.org/html/2605.22064v2) | Supported; external benchmark superiority claims are not used. |
| `doan2021phomt` | PhoMT contains 3.02M English-Vietnamese sentence pairs. | Abstract, PDF page 1; dataset discussion. | [Primary source](https://aclanthology.org/2021.emnlp-main.369.pdf) | Supported; it is not the source of the submitted system training corpus. |
| `ngo2022mtet` | MTet contains 4.2M training pairs and introduces EnViT5. | Abstract, PDF page 1; dataset/model sections. | [Primary source](https://arxiv.org/pdf/2210.05610) | Supported; 6.2M refers to the combined resource, not MTet alone. |
| `voita2019context` | Context-aware MT examines deixis, ellipsis and lexical cohesion. | Abstract and discourse-test descriptions in the paper. | [Primary source](https://aclanthology.org/P19-1116.pdf) | Supported; does not establish document-wide consistency of our block pipeline. |
| `maruf2021survey` | Document-level NMT methods and evaluation are surveyed. | Sections 3 and 4 of the author manuscript; published DOI metadata checked separately. | [Primary source](https://arxiv.org/html/1912.08494v3) | Supported; online 2021 versus print 2022 is documented in VERIFICATION.md. |
| `wang2023docllm` | LLMs have been evaluated as document-level translators. | Abstract and the paper's document-level evaluation description. | [Primary source](https://aclanthology.org/2023.emnlp-main.1036.pdf) | Supported; not evidence that a small independent-block system resolves discourse. |
| `singh2025indictrans2` | Legal-domain adaptation has used QLoRA with IndicTrans2. | Section 4.2, PDF page 3; Table 2 explicitly lists 4-bit quantisation. | [Primary source](https://aclanthology.org/2025.justnlp-main.15.pdf) | Supported; this is English-Hindi, not English-Vietnamese. No cross-task numerical comparison is made. |
| `lorini2026eu` | Legal MT work adapts open-weight LLMs to EU legislation. | Sections 3.2 and 3.3, PDF pages 5-6; Section 4 results. | [Primary source](https://aclanthology.org/2026.eamt-1.20.pdf) | Supported; their EuroLLM-22B is outside the current task size limit and is related work only. |
| `holtzman2020degeneration` | Maximisation-based generation can suffer degenerate repetition. | Sections 2.2, 3 and 5.3. | [Primary source](https://arxiv.org/html/1904.09751v2) | Supported as general generation motivation; our legal-MT repetition rates require local logs. |
| `keskar2019ctrl` | A repetition penalty discounts previously generated tokens. | Section 4.1, Sampling, including the penalised sampling equation. | [Primary source](https://arxiv.org/html/1909.05858v2) | Supported; the local HF implementation and recovery-only use are separately checked in code. |
| `feng2022labse` | LaBSE provides cross-lingual sentence embeddings used here for similarity filtering. | Abstract and multilingual embedding formulation. | [Primary source](https://aclanthology.org/2022.acl-long.62.pdf) | Supported for the embedding method; our flagged-pair-only filtering and threshold are established by local data preparation records, not this source. |
| `indexteam` | Index-Translate provides a 2B translation model. | Model-family description in the official report and pinned model card. | [Primary source](https://arxiv.org/html/2609.40181v2) | Supported for the base model. The model report does not verify our fine-tuned Index comparison scores. |
| `post2018sacrebleu` | Metric configuration must be reported for reproducible BLEU scores. | Paper's reproducibility discussion; local scores contain full SacreBLEU signatures. | [Primary source](https://aclanthology.org/W18-6319.pdf) | Supported; local version is 2.6.0. LF restoration is now explicit. |
| `popovic2015chrf` | chrF uses a harmonic mean of order-averaged character precision and recall. | Section 2, equation 1, PDF page 1. | [Primary source](https://aclanthology.org/W15-3049.pdf) | Supported; beta=2, effective order and whitespace exclusion are our SacreBLEU defaults, not attributed to the original paper's experiment settings. |
| `papineni2002bleu` | BLEU combines modified n-gram precision with a brevity penalty. | Sections 2.2.2 and 2.3, PDF page 5. | [Primary source](https://aclanthology.org/P02-1040.pdf) | Supported; factor 100 converts the original 0-1 score to reported 0-100 values. |
| `koehn2004significance` | Bootstrap resampling estimates BLEU uncertainty using corpus rescoring. | Section 5, PDF pages 5 onward. | [Primary source](https://aclanthology.org/W04-3250.pdf) | Supported; we resample whole documents rather than independent sentences and report percentile intervals, not invented p-values. |

## Verified local findings

- Initial source documents: 575; dev source documents: 41. Final real-document IDs (orig/paragraph rows only): 612 = 571 retained initial train + 41 dev. Four initial IDs are absent after filtering: `Quyen_dan_su/440`, `So_huu_tri_tue/140`, `So_huu_tri_tue/167`, `Trach_nhiem_hinh_su/165`. Synthetic phrase/term identifiers are not counted as source documents.
- Final training rows: 82,203. Composition now aggregates terminology as one category. Training prompts using glossary hints: 19,495 / 82,203 = 23.7%.
- Glossary: 442 entries. Every entry has `audit_status=needs_human_semantic_review` in the source provenance. Historical `status=verified` is not sufficient evidence of expert validation. Revised manuscript makes only the automatic co-occurrence claim.
- Alignment queue: 237 flagged views, 158 removal decisions. The source release explicitly warns that 141 removed views were not individually read by a human. Removed the claim that all 158 were manually reviewed.
- The previous 41,431 tone-change count was not traced to a matching source statistic. Removed the numerical count rather than presenting it as independently verified. The normalisation rules remain supported by code.
- Hy-MT2 final public predictions match BLEU 70.87/47.44 and chrF 84.48/79.50 after restoring official LF line endings. Saved final run: `v33-fast16-a-penalty100-public-20261010_public`.
- Final run logs record 7,229.360 / 4,870.592 seconds and 4,957,004,288 / 5,253,570,560 peak GPU bytes (decimal GB in the manuscript).
- Recovery units: EN-VI 138/2,899 (4.76%), VI-EN 326/3,086 (10.56%). Recovery backend requests are different quantities: 836/1,443. Unresolved units are 26/2,899 (0.90%) and 208/3,086 (6.74%).
- Final runtime log: torch 2.11.0+cu128, Transformers 5.8.1, tokenizers 0.22.2; SacreBLEU 2.6.0.

## Findings that remain unresolved

- The initial main-table baselines and Index comparison rows have no matching raw prediction sets in the searched local records. Their numeric values are preserved from the original manuscript but are not independently certified by this audit.
- Locally archived Index v3.3 public scores are 67.58/46.20, and the older archived release has 63.95/43.42; neither matches the manuscript's Index 62.59/41.76. Different checkpoints or inference configurations could explain this, so these archives must not be silently substituted.
- The previous revision prematurely linked the available Index training configuration to the manuscript comparison. The PDF now explicitly separates this archived configuration from the unverified comparison association.
- A three-seed baseline mean/standard deviation requires the three actual seeds, per-seed metrics and predictions. These were not located. Bootstrap over documents is not a substitute for seed variability.
- No paired bootstrap statement is made for main-table base-versus-pipeline or base-versus-fine-tuned comparisons without the corresponding predictions. Grouping A/B/C can be checked because all saved outputs are available.
- Timing has one recorded run per configuration; no repeated-run speed confidence interval is inferred.
- Author email addresses remain unavailable.

## Newline reproducibility correction

The working checkout public CSV has MD5 `b94c617ec2cdfa86eb1ae181ba96bf21`; three archived official copies have MD5 `d86e055d3f84436ae07e6c27fe686bcc`. Their only byte differences are CRLF/LF. Reading the working CSV with universal newlines produces exactly the same 50 rows and all fields as reading the archived official CSV without conversion. `tools/score_public.py` and the reproduction script now use that policy, matching the existing prediction `read_text` behaviour. No lexical or Unicode normalisation was introduced, and the dataset file itself was not modified.

## Reproduction

```powershell
& 'C:/Users/trand/Documents/vlsp-pipeline-main/.venv/Scripts/python.exe' paper/acl/verification/reproduce_statistics.py --csv dataset/public_test/en-vi-public_test.csv --abc-root 'C:/Users/trand/Documents/vlsp-pipeline-main/results/public_test_20261004' --final-pred 'C:/Users/trand/Documents/VLSP26-FINAL/run/kout/predictions_public' --output paper/acl/verification/statistics.json
```

Requires `sacrebleu==2.6.0` and NumPy. Seed 42; 1,000 paired whole-document resamples for A/B/C; 2,000 whole-document resamples for final-system score intervals. The latter are uncertainty of the corpus score, not evidence of improvement over a missing baseline. Each interval is conditional on these 50 development documents and the saved checkpoint; it does not measure private-test performance or training-seed uncertainty.

## Primary local evidence files

- `C:/Users/trand/Documents/VLSP26/Final_Data/data_release/release_report.json` and the train/dev JSONL files.
- `C:/Users/trand/Documents/Index-Translate-2B-Legal-LoRA-v33-FULL/training/data/index_data_v33/train.jsonl`, `build_report_v33.json`, `results/run_manifest.json`, and public-score JSON files.
- `C:/Users/trand/Documents/HYMT2_V33_FINETUNE_20261009/data/built/build_report.json`, `data/source/glossary_provenance.jsonl`, training code and trainer states.
- `C:/Users/trand/Documents/vlsp-pipeline-main/results/public_test_20261004/` (all three saved prediction sets).
- `C:/Users/trand/Documents/vlsp-pipeline-main/delivery/v33_a_penalty100_public/` (saved final logs), and `C:/Users/trand/Documents/VLSP26-FINAL/run/kout/predictions_public/` (saved final predictions).

## Full reproducibility identifiers

- Hy-MT2 base revision: `9a341cd1b679d3efd23b46e847b01745a71ed792`.
- Index base revision: `a516854233b170140b57d36b48d7148c0edebabb`.
- Hy-MT2 adapter SHA-256: `30b5fc68a31c3de3d722a6dee5484aaf98c4f6b9d7b41fb55ac70f6d23a3ba89`.
- Archived Index continuation: rank 64, alpha 128, dropout 0.05; fp16 DDP on two T4s; global accumulation 16; max length 4096; one epoch, 5138 steps; LR 5e-5, 25 warm-up steps, cosine to 10%; PagedAdamW8bit, weight decay 0.01, clipping 1.0, seed 42. This describes the archive, not a verified match to the paper comparison row.

## Recomputed numerical results

| Comparison | EN-VI delta BLEU [95% CI] | VI-EN delta BLEU [95% CI] |
|---|---|---|
| B-A | +0.20 [+0.05, +0.37] | +0.25 [-0.06, +0.68] |
| A-C | +1.01 [+0.60, +1.32] | +1.14 [+0.20, +1.97] |
| B-C | +1.21 [+0.75, +1.52] | +1.39 [+0.61, +2.20] |

Final BLEU intervals: EN-VI [65.68, 74.24]; VI-EN [42.30, 50.99]. These are not paired baseline-comparison intervals. The saved final adapter SHA-256 was recomputed from both the archived adapter and the current repo adapter and matched exactly. Actual SFT JSONL rows and hinted rows were counted again: 82,203 and 19,495. All 442 glossary provenance entries carry the pending human-semantic-review audit status.

Memory clarification: `HFBackend.peak_vram()` returns `torch.cuda.max_memory_allocated()`. Manuscript memory figures now say peak allocated GPU memory, in decimal GB. These do not include all CUDA/context/reserved-device memory.

## What the current statistics still cannot establish

1. A valid paired interval for the main-table base-versus-pipeline comparisons needs the exact paired outputs for each row.
2. The reported Hy-MT2 baseline standard deviations require the actual three seeds and their scores; they cannot be reconstructed from a mean and standard deviation.
3. Separating pipeline gains from sampling-to-greedy changes requires a matched greedy baseline; separating fine-tuning from data cleaning requires additional controlled training runs. The present CPU audit does not create these experiments.
4. Repeated speed measurements are needed before quantifying timing variance or claiming a small speed advantage.
5. Corpus-score intervals, paired score-difference intervals and training-seed standard deviations measure different uncertainties and are kept separate.

## Follow-up: Documents-wide source search

A Documents-wide scan of 6,353 text/log/notebook files, 20 relevant ZIP archives and 143 historical Git blobs did not locate primary scoring evidence matching the main-table Index 62.59/41.76 or the baseline rows. Four different archived Index runs were located, including glossary 67.69/45.86 and historical pilot step 374 at 60.05/41.10. The target pair already appears in the paper imported by commit 97e5442 on 2026-10-15. See [search report](DOCUMENTS_SEARCH.md).

## Author decision (2026-10-27)

The author requests retaining the main-table baseline and Index comparison values exactly as in the original manuscript. A direct row comparison confirms they are unchanged. No further search or replacement of those values is planned; retaining them is an author decision, not a new certification of the missing run provenance.
