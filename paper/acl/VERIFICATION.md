# Paper verification (2026-10-27)

## Reference audit

All 23 entries in the previous bibliography were checked. The revised text cites 20; three tangential model citations were removed while condensing Related Work. No unchecked source or invented metadata was added. Canonical metadata was read directly from ACL Anthology BibTeX, arXiv records, the official model card and publisher-deposited Crossref metadata.

| Key | Primary source | Result |
|---|---|---|
| `doan2021phomt` | [Source](https://aclanthology.org/2021.emnlp-main.369/) | Retained. Imported title, all authors, year, venue and pages from the official ACL Anthology BibTeX; retained DOI where supplied. |
| `voita2019context` | [Source](https://aclanthology.org/P19-1116/) | Retained. Imported title, all authors, year, venue and pages from the official ACL Anthology BibTeX; retained DOI where supplied. |
| `wang2023docllm` | [Source](https://aclanthology.org/2023.emnlp-main.1036/) | Retained. Imported title, all authors, year, venue and pages from the official ACL Anthology BibTeX; retained DOI where supplied. |
| `singh2025indictrans2` | [Source](https://aclanthology.org/2025.justnlp-main.15/) | Retained. Imported title, all authors, year, venue and pages from the official ACL Anthology BibTeX; retained DOI where supplied. |
| `lorini2026eu` | [Source](https://aclanthology.org/2026.eamt-1.20/) | Retained. Imported title, all authors, year, venue and pages from the official ACL Anthology BibTeX; retained DOI where supplied. |
| `feng2022labse` | [Source](https://aclanthology.org/2022.acl-long.62/) | Retained. Imported title, all authors, year, venue and pages from the official ACL Anthology BibTeX; retained DOI where supplied. |
| `post2018sacrebleu` | [Source](https://aclanthology.org/W18-6319/) | Retained. Imported title, all authors, year, venue and pages from the official ACL Anthology BibTeX; retained DOI where supplied. |
| `popovic2015chrf` | [Source](https://aclanthology.org/W15-3049/) | Retained. Imported title, all authors, year, venue and pages from the official ACL Anthology BibTeX; retained DOI where supplied. |
| `papineni2002bleu` | [Source](https://aclanthology.org/P02-1040/) | Retained. Imported title, all authors, year, venue and pages from the official ACL Anthology BibTeX; retained DOI where supplied. |
| `koehn2004significance` | [Source](https://aclanthology.org/W04-3250/) | Retained. Imported title, all authors, year, venue and pages from the official ACL Anthology BibTeX; retained DOI where supplied. |
| `alves2024tower` | [Source](https://arxiv.org/abs/2402.17733) | Retained. Verified title, all authors and arXiv identifier against the primary arXiv record. |
| `luo2025lmt` | [Source](https://arxiv.org/abs/2511.07003) | Verified; removed from shortened Related Work. Verified title, all authors and arXiv identifier against the primary arXiv record. Corrected the title from Beyond English to NiuTrans.LMT; this is a multilingual suite, not a bilingual-only model. |
| `zheng2025hunyuanmt` | [Source](https://arxiv.org/abs/2509.05209) | Verified; removed from shortened Related Work. Verified title, all authors and arXiv identifier against the primary arXiv record. |
| `zheng2026hymt2` | [Source](https://arxiv.org/abs/2605.22064) | Retained. Verified title, all authors and arXiv identifier against the primary arXiv record. |
| `nllb2022` | [Source](https://arxiv.org/abs/2207.04672) | Verified; removed from shortened Related Work. Verified title, all authors and arXiv identifier against the primary arXiv record. |
| `ngo2022mtet` | [Source](https://arxiv.org/abs/2210.05610) | Retained. Verified title, all authors and arXiv identifier against the primary arXiv record. |
| `keskar2019ctrl` | [Source](https://arxiv.org/abs/1909.05858) | Retained. Verified title, all authors and arXiv identifier against the primary arXiv record. Corrected erroneous year 1909 to 2019 and repaired identifier 1909.05858. |
| `hu2022lora` | [Source](https://arxiv.org/abs/2106.09685) | Retained. Verified title, all authors and arXiv identifier against the primary arXiv record. ICLR 2022 venue/year verified against microsoft/LoRA official README citation (OpenReview itself returned a browser challenge). |
| `holtzman2020degeneration` | [Source](https://arxiv.org/abs/1904.09751) | Retained. Verified title, all authors and arXiv identifier against the primary arXiv record. ICLR 2020 is confirmed by the arXiv publication comment. |
| `xu2024alma` | [Source](https://arxiv.org/abs/2309.11674) | Retained. Verified title, all authors and arXiv identifier against the primary arXiv record. ICLR 2024 is confirmed by the arXiv acceptance comment. |
| `maruf2021survey` | [Source](https://doi.org/10.1145/3441691) | Retained. Verified authors, title/subtitle, volume 54(2), pages 1-36 and DOI against Crossref publisher-deposited metadata. Online publication: 2021-03-05; print issue: 2022-03-31. Cite the print year 2022. The citation key is retained for compatibility. |
| `indexteam` | [Source](https://arxiv.org/abs/2609.40181) | Retained. Replaced undated model-name-only reference with the official technical report linked in the pinned Hugging Face model card. All 11 authors, title and 2026 date match arXiv. The model-card checkpoint used in experiments is pinned in the manuscript. |
| `vlsp2026` | [Source](https://vlsp.org.vn/vlsp2026/eval/mt_challenge) | Retained. Official challenge webpage, not a proceedings paper. Used the actual page title and a corporate author; it explicitly recommends public-test scoring for system selection. |

Supporting sources: [official LoRA citation](https://github.com/microsoft/LoRA#how-to-cite), [Crossref record](https://api.crossref.org/works/10.1145/3441691), [pinned Index model card](https://huggingface.co/IndexTeam/Index-Translate-2B/blob/a516854233b170140b57d36b48d7148c0edebabb/README.md).

## Formula and implementation audit

- Loss: reformulated over unmasked label positions. Hy-MT2 `scripts/train_v33/train_lora.py` supervises the assistant separator through response EOS; Index `training/code/train_base.py` masks the entire generation prompt. The previous translation-only wording did not cover both.
- LoRA update: B is d by r, A is r by k, so W0 + (alpha/r) BA is dimensionally correct; ordinary LoRA is used, not rank-stabilised LoRA.
- Sequence length: counts the actual tokenised chat template. Removed an assumed five-token constant that is not valid across both templates and can double-count EOS under the previous target definition.
- BLEU: retained four equal n-gram weights and BP; added factor 100 for reported 0-100 scores. Defined precision as fractions and noted implementation handling of zero matches/empty outputs.
- chrF: retained harmonic mean of order-averaged precision/recall; added factor 100. Confirmed order 6, word order 0, beta 2, whitespace excluded and default effective-order behaviour against SacreBLEU implementation.
- Bootstrap delta and percentile interval: checked against the primary methodological source. Saved A/B/C outputs were rescored and 1,000 paired document resamples recomputed with seed 42; final-system score intervals use 2,000 resamples. Main-table baseline comparisons still lack matching saved outputs; no paired significance claim is made for them.
- LoRA parameter count: 32 * 64 * (2*(2048+2048) + 2*(2048+512) + 3*(2048+6144)) = 77,594,624. Adapter/(base+adapter) = 4.152387%; adapter/base = 4.332280%. These are distinct denominators.
- Added output-budget equation from `docmt/steps/s05_budget.py`: a=1.25 only for VI-to-EN first-pass prose; protected units/full-budget recovery use a=2.5. Actual generation is also bounded by prompt, margin and sequence ceiling.

## Implementation sources and remaining limits

- Pipeline: `docmt/steps/s02_blocks.py`, `s03_pack.py`, `s05_budget.py`, `s06_parse_output.py`, `s07_validate_unit.py`, `s08_recovery.py`, `s09_select.py`; `configs/hy_mt_vlsp26.json`.
- Hy-MT2 training: `C:/Users/trand/Documents/HYMT2_V33_FINETUNE_20261009/scripts/train_v33/train_lora.py` and `kernel_train_v33.py`.
- Index final continuation: `C:/Users/trand/Documents/Index-Translate-2B-Legal-LoRA-v33-FULL/training/code/train_v3.py`, `train_base.py`, `training/kaggle/full_kernel.py` and `adapter/adapter_config.json`. Effective batch 16 is GLOBAL in Index, versus 32 in Hy-MT2; maximum lengths are 4096 and 3072, respectively.
- Index appendix settings describe the final continuation command and adapter configuration; they do not invent a complete warm-start history.
- Preserve all main sections and appendix sections A-C, all reported BLEU/chrF and cost values, and both original figure files. No GPU experiments or training runs were performed.
- Official private-test scores/rankings and human legal evaluation are not available in this manuscript.
- Author email addresses remain to be supplied. The document-count transition has now been verified directly: 575 initial train + 41 development - 4 removed train documents = 612 real source documents. Synthetic phrase/term IDs are excluded from the document count.

Index training prompt, read directly from `training/code/train_base.py` (source/target language names are Chinese):

```text
请将以下{source_language}文本翻译为{target_language}，直接输出翻译结果，不要进行任何解释。

{source_text}
```

The Index trainer always constructs this default instruction; mixed glossary prompts in the paper are therefore explicitly attributed to Hy-MT2.

## Build validation

The final PDF compiles to seven pages, including all references and appendix sections A-C. No overfull boxes, undefined citations/references or missing glyphs were reported. All fonts are embedded. Original experimental score/time entries are retained. Terminology categories in the training composition table are aggregated without changing the total. The unmodified ACL style files and both figure PDFs are preserved.

## Deeper follow-up verification

See [claim-level audit](verification/CLAIM_AUDIT.md), [recomputed statistics](verification/statistics.json) and [reproduction script](verification/reproduce_statistics.py). These supersede the earlier statement that the archived Index settings were associated with the reported comparison; that association remains unresolved. The latest verification also corrects the glossary-review and manual-alignment wording, resolves the document-count transition and adds explicit LF scoring. No model inference or training was run.
