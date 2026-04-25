ABALATION TABLES:

TABLE 1

| Model Variant | Task 1 (Drug Recommendation) |  |  | Task 2 (In-Hospital Mortality) |  | Task 3 (Hospital Readmission) |  |
| :---: | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
|  | **AUPRC** | **Precision@k** | **Recall@k** | **AUPRC** | **AUROC** | **AUPRC** | **AUROC** |
| **Retain Full** |  |  |  |  |  |  |  |
| **Retain No Alpha** |  |  |  |  |  |  |  |
| **Retain No Beta** |  |  |  |  |  |  |  |
| **Retain neither** |  |  |  |  |  |  |  |

Retain with full components vs using no alpha (uniform) vs no beta (identity) vs neither

Table 2

| Model Variant | Task 1 (Drug Recommendation) |  |  | Task 2 (In-Hospital Mortality) |  | Task 3 (Hospital Readmission) |  |
| :---: | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
|  | **AUPRC** | **Precision@k** | **Recall@k** | **AUPRC** | **AUROC** | **AUPRC** | **AUROC** |
| **Retain Base** |  |  |  |  |  |  |  |
| **Retain Focal Loss** |  |  |  |  |  |  |  |
| **Retain Bidirectional** |  |  |  |  |  |  |  |
| **Retain Both** |  |  |  |  |  |  |  |

Retain with our normal loss vs using focal loss vs using bidirectionality 

BASELINES

| Model Variant | Task 1 (Drug Recommendation) |  |  | Task 2 (In-Hospital Mortality) |  | Task 3 (Hospital Readmission) |  |
| :---: | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
|  | **AUPRC** | **Precision@k** | **Recall@k** | **AUPRC** | **AUROC** | **AUPRC** | **AUROC** |
| **Baseline** |  |  |  |  |  |  |  |
| **RETAIN** |  |  |  |  |  |  |  |
| **Improvement** |  |  |  |  |  |  |  |
| **Retain Both** |  |  |  |  |  |  |  |

