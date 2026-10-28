# Experiment-derived architectural decisions

## Evidence, not claims
- Bail Stage-0 MLP AUC .9919 vs graph GCN .9311: feature-dominated; potential leakage; **negative control**.
- Credit Stage-0 GCN AUC .7070 vs MLP .6466: provisional graph signal (rewiring collapse confound).
- Bail A2 F1 .928 but same as random-mask control; important predictor ≠ important explainer.
- The 90-run mask strategies have near-constant Bail scores in all 45 runs, Credit in 33/45; Credit learned-minus-random AUC was negative.
- Old A0-A5 and later v2 use different protocols: never pool or directly compare.

## Stages
B0: predictive GCN vs MLP (no edge mask).
B1: freeze predictor, score edges, learn soft mask; evaluate hard top-k vs random/degree.
B2: predictor fairness-constrained ablation; fairness-targeted explanation parity must be measured separately.
B3: perturbation robustness; compare unchanged-edge masks **only if prediction remains unchanged**.
B4: combine validated fairness and explanation robustness, with utility tolerance.
B5: add teacher snapshot, task-indexed exemplar replay, proximal regularization; measure forgetting across sequential tasks.
B6: optionally use federated simulated clients with FedAvg baseline, then FL personalization/distillation; check non-IID clients and fairness gaps.

## Proposed research hypothesis
A distillation-and-replay-aware, fairness-constrained GNN with a separately validated edge explainer can outperform naïve joint mask training on **faithfulness and retention** without requiring accuracy degradation. This is a hypothesis, not a result.

## Novelty caution
Fairness regularization, edge masks, consistency and distillation are established ideas. Contribution must be established via a particular *mathematically defensible coupling*, substantive benchmark gains, and stringent controls. A robust direction-aware geometric metric is not yet implemented and GeoXGNN is prior work.

## Data and protocol
Do not invent reported experiments. Add vetted original datasets and splits locally; exclude sensitive labels from predictive features according to each baseline's published protocol. Report group distribution, class prevalence, thresholds chosen on validation only, equal budgets, standard deviations and paired seed comparisons. For genuine FL: clients must have disjoint training-node supervision, never validation/test labels, while graph boundary visibility must be declared.
