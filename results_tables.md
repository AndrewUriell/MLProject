# Results Tables

> **Bold** = best per column · <u>Underline</u> = second best per column
> *(In plain Markdown viewers, underline is shown as `<u>text</u>` — apply manually in your editor)*

---

## Table 1 — Attention Ablation Study (α / β)

| Model Variant | Drug AUPRC | Drug P@10 | Drug P@20 | Drug R@10 | Drug R@20 | Mort. AUPRC | Mort. AUROC | Read. AUPRC | Read. AUROC |
|---|---|---|---|---|---|---|---|---|---|
| RETAIN Full | **0.653** | **0.824** | **0.717** | **0.263** | **0.443** | <u>0.205</u> | <u>0.652</u> | <u>0.345</u> | <u>0.578</u> |
| RETAIN No Alpha | 0.602 | 0.786 | 0.678 | 0.246 | 0.414 | 0.179 | 0.633 | 0.342 | 0.552 |
| RETAIN No Beta | <u>0.642</u> | <u>0.814</u> | <u>0.711</u> | <u>0.259</u> | <u>0.438</u> | **0.211** | **0.656** | 0.333 | <u>0.583</u> |
| RETAIN Neither | 0.633 | 0.802 | 0.699 | 0.254 | 0.430 | 0.206 | 0.653 | **0.368** | **0.592** |

*RETAIN with both attention mechanisms vs. removing visit-level attention (No Alpha), code-level attention (No Beta), or both (Neither).*

---

## Table 2 — Architectural Modification Study

| Model Variant | Drug AUPRC | Drug P@10 | Drug P@20 | Drug R@10 | Drug R@20 | Mort. AUPRC | Mort. AUROC | Read. AUPRC | Read. AUROC |
|---|---|---|---|---|---|---|---|---|---|
| RETAIN Base | <u>0.653</u> | **0.824** | **0.717** | **0.263** | **0.443** | **0.205** | **0.652** | <u>0.345</u> | <u>0.578</u> |
| RETAIN Focal Loss | 0.637 | 0.806 | 0.703 | 0.256 | 0.434 | 0.191 | <u>0.648</u> | **0.359** | **0.593** |
| RETAIN Bidir. | **0.654** | <u>0.823</u> | <u>0.716</u> | <u>0.263</u> | <u>0.443</u> | <u>0.201</u> | 0.632 | 0.310 | 0.549 |
| RETAIN Both | 0.647 | 0.819 | 0.713 | 0.261 | 0.441 | 0.184 | 0.604 | 0.341 | 0.577 |

*RETAIN base vs. adding Focal Loss, Bidirectional α-GRU, or both modifications.*

---

## Table 3 — Baseline Comparison

| Model | Drug AUPRC | Drug P@10 | Drug P@20 | Drug R@10 | Drug R@20 | Mort. AUPRC | Mort. AUROC | Read. AUPRC | Read. AUROC |
|---|---|---|---|---|---|---|---|---|---|
| Baseline | <u>0.611</u> | <u>0.786</u> | <u>0.682</u> | <u>0.249</u> | <u>0.420</u> | **0.239** | <u>0.648</u> | **0.365** | **0.607** |
| RETAIN | **0.653** | **0.824** | **0.717** | **0.263** | **0.443** | <u>0.205</u> | **0.652** | <u>0.345</u> | <u>0.578</u> |
| Δ Improvement | +0.042 | +0.038 | +0.035 | +0.014 | +0.023 | -0.035 | +0.004 | -0.021 | -0.029 |
| % Improvement | +4.200% | +3.870% | +3.470% | +1.410% | +2.310% | -3.450% | +0.420% | -2.060% | -2.920% |

*RNN / Transformer baselines vs. RETAIN (full). Δ and % rows show absolute and relative improvement.*
