ABALATION TABLES:

Table 1:

|  Model Variant | Task 1 (Drug Recommendation) |  |  |  |  | Task 2 (In-Hospital Mortality) |  | Task 3 (Hospital Readmission) |  |
| :---: | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
|  |  **AUPRC** | **Precision@k** |  | **Recall@k** |  |  **AUPRC** |  **AUROC** |  **AUPRC** |  **AUROC** |
|  |  | **k \= 10** | **k \= 20** | **k \= 10** | **k \= 20** |  |  |  |  |
| **Retain Full** | 0.6527994253989591 | 0.8244883556810163 | 0.7168313338038109 | 0.2630178579843836 | 0.44333200466841677 | 0.20470651404516693 | 0.6521749833829793 | 0.34454669008250144 | 0.5780763549596636 |
| **Retain  No Alpha** |  |  |  |  |  |  |  |  |  |
| **Retain No Beta** |  |  |  |  |  |  |  |  |  |
| **Retain Neither** |  |  |  |  |  |  |  |  |  |

Retain with full components vs using no alpha (uniform) vs no beta (identity) vs neither

Table 2

|  Model Variant | Task 1 (Drug Recommendation) |  |  |  |  | Task 2 (In-Hospital Mortality) |  | Task 3 (Hospital Readmission) |  |
| :---: | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
|  |  **AUPRC** | **Precision@k** |  | **Recall@k** |  |  **AUPRC** |  **AUROC** |  **AUPRC** |  **AUROC** |
|  |  | **k \= 10** | **k \= 20** | **k \= 10** | **k \= 20** |  |  |  |  |
| **Retain Base** | 0.6527994253989591 | 0.8244883556810163 | 0.7168313338038109 | 0.2630178579843836 | 0.44333200466841677 | 0.20470651404516693 | 0.6521749833829793 | 0.34454669008250144 | 0.5780763549596636 |
| **Retain Focal Loss** |  |  |  |  |  |  |  |  |  |
| **Retain Bidirectional** |  |  |  |  |  |  |  |  |  |
| **Retain Both** |  |  |  |  |  |  |  |  |  |

Retain with our normal loss vs using focal loss vs using bidirectionality 

BASELINES

|  Model Variant | Task 1 (Drug Recommendation) |  |  |  |  | Task 2 (In-Hospital Mortality) |  | Task 3 (Hospital Readmission) |  |
| :---: | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
|  |  **AUPRC** | **Precision@k** |  | **Recall@k** |  |  **AUPRC** |  **AUROC** |  **AUPRC** |  **AUROC** |
|  |  | **k \= 10** | **k \= 20** | **k \= 10** | **k \= 20** |  |  |  |  |
| **Baseline** | 0.6107732045319351 | 0.7857445306986592 | 0.682180663373324 | 0.2488819718346738 | 0.4201615860953051 | 0.23929075375589792 | 0.6479899559341228 | 0.365115203138733 | 0.6072548574025679 |
| **Retain**  | 0.6527994253989591 | 0.8244883556810163 | 0.7168313338038109 | 0.2630178579843836 | 0.44333200466841677 | 0.20470651404516693 | 0.6521749833829793 | 0.34454669008250144 | 0.5780763549596636 |
| **ΔImprov** | 0.0420262209 | 0.038743825 | 0.0346506704 | 0.0141358861 | 0.0231704186 | \-0.0345842397 | 0.00418502745 | \-0.0205685131 | \-0.0291785024 |
| **%Improv** | 4.2% | 3.87% | 3.47% | 1.41% | 2.31% | \-3.45% | 0.42% | \-2.06% | \-2.92% |

