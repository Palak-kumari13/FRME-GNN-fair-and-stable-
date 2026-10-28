# FRME-GNN v2.1 — Graph bridge and module fallback specification

## Pipeline

```
Node features + optional edge list
  → leakage/feature audit (required external data protocol)
  → graph bridge
     ├─ original graph, if structurally valid
     ├─ feature kNN, if original fails
     └─ feature-only MLP, if graph construction fails
  → validation-only graph signal gate (graph GCN vs MLP)
     ├─ graph selected → train GCN → train edge explainer → mask diagnostic gate
     └─ MLP selected → skip all edge explanation claims
  → fairness/robustness experimental losses
  → prior-task distillation/replay (experimental)
  → final metrics and chosen-route audit trail
```

## Module fallback design (implemented vs planned)
| Module | Failure | Fallback | Status |
|---|---|---|---|
| Graph construction | Missing/invalid/fragmented edge list | kNN then feature only | Implemented |
| Graph signal | GCN lacks prespecified validation-AUROC gain over MLP | MLP | Implemented |
| Mask | Nearly constant scores / order dependence | Suppress evidence claim, retain predictor | Implemented integrity status only; causal fidelity gate still planned |
| Fairness | Constraint breaches preset utility floor | Restore acceptable checkpoint / tune Lagrange multiplier | Planned |
| Robustness | Drop under admitted graph edits | Restore stable checkpoint / retrain augmentation | Planned |
| Past learning | Exceeds forgetting budget on old validation tasks | Restore teacher / expand rehearsal memory | Planned |
| Federated | Drift/divergence or unstable clients | Roll back round, FedProx or adaptive server step | Planned |

## Mandatory caution
A graph can be topologically well-formed but carry no useful predictive signal. Similarly, a masked predictor can give good AUC despite meaningless edge scores. Graph bridge is a standard graph structure learning / construction idea, and fallback routing alone is **not an established novel contribution**. To claim a new method requires a precise algorithm and strong comparisons at fixed preprocessing/splits against GCN, MLP, FairGNN, PGExplainer, and GSL baselines. Mask integrity passing is necessary, not sufficient: test planted-edge recovery, sufficiency/comprehensiveness, matched random and degree controls, tie sensitivity, and seed variation.

This version uses small transductive synthetic node data only and does not include the full external benchmark loaders.
