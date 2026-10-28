# FRME-GNN v2.2 — integrated stability, fairness and fallback

## What is inherited from published work vs. what is new engineering

| Published foundation | Verified published idea | Our implementation/difference | Claim limits |
|---|---|---|---|
| FairGNN (WSDM 2021) | Adversarial fairness with sensitive attribute estimation | **Not an exact FairGNN reproduction**: differentiable DP/EO proxy penalty and explicit validation-time fairness-vs-AUC gate | Comparison to FairGNN requires faithful external reproduction |
| PGExplainer (NeurIPS 2020) | Parameterized edge-scoring explanations | Small inductive scorer trained on soft masks; controls for near-constant scores, order invariance, matched random/degree baselines | Not equivalent to PGExplainer training objective; screening is not evidence of causal importance |
| GeoXGNN (IJCAI 2026) | Direction-aware geometry to stabilize graph explanations | Simple consistency regularization for surviving edges and joint prediction robustness gate | **Does not implement GeoXGNN's anisotropic local metric** |
| Graph structure learning (GSL) | Learning/reconstructing useful graph topology | Original-edge audit, cosine kNN bridge, validation-only GNN-vs-MLP utility gate | A kNN fallback is not novel and does not imply graph usefulness |
| FedProx and continual distillation/replay | Non-IID stabilization, retention | Existing simulation utilities + frozen teacher, old-task rehearsal and retention gate | Not privacy-preserving federated learning; not a proven anti-forgetting method |

## Actual v2.2 change relative to v2.1

1. `frme/controller.py`: explicit `GateDecision` records and fail-closed checks.
2. **Fairness gate** requires measured validation DP and EO under limits **and** bounded AUC loss relative to the reference model. Undefined metrics result in rejection, not artificial zeros.
3. **Stability gate** perturbs the graph several times and tests prediction preservation / probability drift on validation nodes, compared with the reference.
4. **Mask gate** abstains if edge scores are nearly constant, permutation consistency fails, or matched random masking is not clearly worse in sufficiency. Degree baseline and further fidelity studies remain necessary.
5. **Retention gate** compares old-task validation AUC before/after sequential learning; failed retention restores old checkpoint.
6. `run_integrated.py`: runs bridge → quality route → fair-and-stable candidate evaluation → explanation gate → continual retention gate → holdout evaluation.

## Critical methodological cautions

- These are **prototype validation gates**, not per-module mathematical guarantees. A returned reference checkpoint may fail fairness or robustness itself; report that limitation explicitly.
- `test_metrics` is computed after decisions. Thresholds must be selected on validation data or inner CV, never test.
- A candidate's *prediction* robustness does not certify explanation robustness. `evaluate_explainer` provides a preliminary common-edge stability measurement; more matched-seed robustness analysis is required.
- The supplied toy graph uses label-correlated connectivity; it is not suitable for comparing against genuine published dataset results.
- The graph pipeline is transductive and kNN can use test-node **features**, not test labels. Specify the intended protocol and forbid all leakage in actual experiments.
- Fairness proxies are computed using known sensitive labels. Missing sensitive attributes and multi-group fairness need separate treatment.
- Graph rebuilding or fallback changes the object being explained. Always log graph provenance.
- The baseline checkpoint can be unfair: rollback is a reliability choice, **not** a fairness guarantee. A deployment-grade solution needs abstention if no certified candidate exists.

## Next benchmark work (not implemented)

Use frozen official Bail, Credit, German splits and compatible FairGNN, PGExplainer, GeoXGNN baselines. Compare controlled B0 bridge off/on, B1 mask off/on, B2 fairness, B3 robustness, B4 both, B5 teacher/replay, B6 joint stack, five paired seeds with bootstrap CIs. Include feature-only MLP, rewired graph, random/degree masking, planted motif, prediction-preserving perturbations, OOD and cross-task retention. Require utility–fairness–fidelity–stability Pareto improvements before a novelty claim.

## References

- FairGNN: *Say No to the Discrimination: Learning Fair Graph Neural Networks with Limited Sensitive Attribute Information*, WSDM 2021.
- PGExplainer: *Parameterized Explainer for Graph Neural Network*, NeurIPS 2020, https://papers.neurips.cc/paper_files/paper/2020/hash/e37b08dd3015330dcbb5d6663667b8b8-Abstract.html
- GeoXGNN: *Local Geometry Improves Explanation Robustness for Graph Neural Networks*, IJCAI 2026, https://ijcai.org/proceedings/2026/62
