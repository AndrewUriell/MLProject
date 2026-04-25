ABALATION TABLES:

Table 1:

|  Model Variant | Task 1 (Drug Recommendation) |  |  |  |  | Task 2 (In-Hospital Mortality) |  | Task 3 (Hospital Readmission) |  |
| :---: | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
|  |  **AUPRC** | **Precision@k** |  | **Recall@k** |  |  **AUPRC** |  **AUROC** |  **AUPRC** |  **AUROC** |
|  |  | **k \= 10** | **k \= 20** | **k \= 10** | **k \= 20** |  |  |  |  |
| **Retain Full** |  |  |  |  |  |  |  |  |  |
| **Retain  No Alpha** |  |  |  |  |  |  |  |  |  |
| **Retain No Beta** |  |  |  |  |  |  |  |  |  |
| **Retain Neither** |  |  |  |  |  |  |  |  |  |

Retain with full components vs using no alpha (uniform) vs no beta (identity) vs neither

Table 2

|  Model Variant | Task 1 (Drug Recommendation) |  |  |  |  | Task 2 (In-Hospital Mortality) |  | Task 3 (Hospital Readmission) |  |
| :---: | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
|  |  **AUPRC** | **Precision@k** |  | **Recall@k** |  |  **AUPRC** |  **AUROC** |  **AUPRC** |  **AUROC** |
|  |  | **k \= 10** | **k \= 20** | **k \= 10** | **k \= 20** |  |  |  |  |
| **Retain Base** |  |  |  |  |  |  |  |  |  |
| **Retain Focal Loss** |  |  |  |  |  |  |  |  |  |
| **Retain Bidirectional** |  |  |  |  |  |  |  |  |  |
| **Retain Both** |  |  |  |  |  |  |  |  |  |

Retain with our normal loss vs using focal loss vs using bidirectionality 

BASELINES

|  Model Variant | Task 1 (Drug Recommendation) |  |  |  |  | Task 2 (In-Hospital Mortality) |  | Task 3 (Hospital Readmission) |  |
| :---: | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
|  |  **AUPRC** | **Precision@k** |  | **Recall@k** |  |  **AUPRC** |  **AUROC** |  **AUPRC** |  **AUROC** |
|  |  | **k \= 10** | **k \= 20** | **k \= 10** | **k \= 20** |  |  |  |  |
| **Baseline** |  |  |  |  |  |  |  |  |  |
| **Retain**  | 0.6527994253989591 | 0.8244883556810163 | 0.7168313338038109 | 0.2630178579843836 | 0.44333200466841677 |  |  |  |  |
| **Improvement** |  |  |  |  |  |  |  |  |  |

