# FRME-GNN — Fair, Robust, Masked and Experience-Aware Graph Learning

**Status: executable research scaffold, NOT benchmark-validated, NOT a reproduction of FairGNN, PGExplainer, GeoXGNN or FedWeIT.**

## What it has
- Lightweight weighted GCN predictor and inductive edge scorer, trained separately to avoid masking being confused with prediction quality.
- Soft retention masks during explainer optimization, discrete exact top-k at evaluation, random- and degree-matched controls, order-equivariance and score-collapse diagnostics.
- Differentiable fairness proxies (demographic parity and equal opportunity), teacher distillation, old-task replay, proximal anchoring, and optional synthetic FedAvg demonstration utilities.
- Prediction-preserving-inspired graph-edit consistency on common edges. This is NOT the directional local-metric algorithm of GeoXGNN.
- Five-seed-capable output and an intentionally small synthetic smoke-test. No dataset files are bundled.

## Install and run
```bash
python -m pip install -r requirements.txt
python -m pytest -q
python run.py --seeds 5 --epochs 100 --robust --out results/synthetic.json
```

## Important limitations
1. Synthetic data includes **no known ground-truth causal edge mask**. Passing this smoke-test does not establish faithful explanations. Implement and verify a planted-motif benchmark first.
2. Graph-signal validation on actual Bail, Credit and German files requires those source files, original preprocessing and frozen benchmark splits; the report alone is insufficient for reproduction.
3. Mask scorer uses a whole-graph transductive setting and is global, not target-node-specific. Node-specific k-hop explanations are a planned extension.
4. This implementation supports a scalar binary sensitive attribute and binary labels. Multiclass/multilabel targets and intersectional fairness need separate protocols.
5. Distillation during synthetic sequential training is restricted to **current** task nodes, while replay uses retained **old** labeled nodes. Current-data distillation is an experimental regularizer, not a rigorous continual-learning guarantee.
6. FedAvg is a simulation utility only; no privacy guarantee, secure aggregation or differential privacy is claimed. FedAvg local rounds initialize from global weights, but lack secure aggregation and do not model real-world client transport or privacy.
7. The robustness module is a first-pass consistency constraint; structural perturbations are not certified or degree-preserving, and its global preservation gate is stricter than a node-specific conditional treatment.
8. Discrete top-k isn't trained end-to-end; continuous mask losses may still fail to learn rankings. Fail if std<1e-4, permutation test fails or learned masks cannot consistently beat random controls.

## Research protocol before claiming gains
1. Reproduce *exactly* the archived runs and repair old CSV 27/29-column mismatch. Do not mix experimental generations.
2. Confirm MLP vs GCN vs degree-preserving rewired GCN. Audit Bail feature leakage.
3. Planted-edge recovery, edge-order invariance, random tie-break sensitivity, score variation across 5 seeds.
4. Run B0 predictor, B1 explainer, B2 + fairness, B3 + robustness, B4 full, then B5 + continual replay/distillation and B6 FL simulation as **separate** ablations.
5. Evaluate AUROC, AUPRC, F1, balanced accuracy, MCC, DP/EO, fidelity KL, sufficiency, comprehensiveness, top-k Jaccard and old-task forgetting with confidence intervals.
6. Explicitly stratify explanation stability by prediction-preserving *per-node* perturbations, and compare with degree/spectral controls.
7. Compare FairGNN, PGExplainer, GeoXGNN using official implementations and fixed splits; avoid claiming this project is those methods.

## Sources
- Local 27 Oct 2026 FRME progress report, bundled as a reference in `docs/`.
- FairGNN (WSDM 2021); PGExplainer (NeurIPS 2020); GeoXGNN (IJCAI 2026, https://ijcai.org/proceedings/2026/62).

See `docs/RESEARCH_PLAN.md` for explicit methodological differences and rejection gates.

## NEW — Graph Reconstruction Bridge + automatic fallback (v2.1)

`frme/graph_bridge.py` adds a **graph construction and audit gate** before the encoder. It checks node count, edge endpoint validity, duplicates, self-loops, isolation rate and weakly connected components. A graph passing these *structural* checks is **not** automatically useful for prediction or explanation.

**Routes:** valid supplied graph → feature cosine kNN construction → feature-only MLP. Dense kNN is capped at 5,000 nodes to prevent accidental quadratic memory blowups. A further **validation AUROC gate** compares a graph model to a feature-only MLP on the same validation nodes and chooses the graph only if it clears the prespecified gain. No test-set metrics are used for routing. For benchmark comparability, record the chosen route, fixed splits, and construction settings, and never reclassify the resulting route as the same model as an unmodified GCN.

```bash
python run.py --seeds 2 --epochs 30 --graph-mode auto --min-graph-gain 0.0 --out results/bridge_demo.json
```

Other controls: `--graph-mode original`, `--graph-mode knn`, `--graph-mode features`. If the MLP route is selected, **edge explanations are disabled rather than fabricated**. Failing mask integrity likewise blocks a positive explanation claim. Robustness, fairness and continual learning remain experimental; do not interpret routing or synthetic tests as research superiority. For kNN, graph construction from node features is transductive; avoid using label or sensitive-group information during construction and audit feature leakage. For inductive evaluation, reconstruct neighbors from training data only and use a predeclared query-time neighbor policy.

**Further hardening required:** fairness-loss rollback, thresholded robustness rollback, teacher-memory rollback, and FL optimizer retry are proposed module-level fallbacks; the present code implements graph/encoder routing and explanation status gating, not the entire controller stack.

## v2.2 — joint fairness/stability controller and continual fallback

Run: `python run_integrated.py --seeds 1 --epochs 35 --out results/integrated.json`

Module gates: `frme/controller.py`. Published-method comparison and scope: `docs/PUBLISHED_METHODS_AND_GATES.md`. The new integrated demo is a **synthetic proof of execution**, not a validated enhancement over FairGNN, PGExplainer or GeoXGNN.
