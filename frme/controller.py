"""Fail-closed modular selection: validation-only gates, auditable safe fallbacks.

Thresholds are research hyperparameters; tune on validation, not test.
"""
from dataclasses import dataclass, asdict
import math
import torch
from .core import metric, perturb_edges, topk_mask

@dataclass
class GateDecision:
    module: str
    passed: bool
    route: str
    reason: str
    measurements: dict
    def report(self): return asdict(self)

def finite(v): return v is not None and isinstance(v,(int,float)) and math.isfinite(v)

def fairness_gate(candidate, reference, *, max_dp=.10, max_eo=.10, max_auc_drop=.03):
    """Both groups/positive labels must be present; missing metrics fail closed."""
    vals={k:candidate.get(k) for k in ('auc','dp_gap','eo_gap')}
    if not all(finite(v) for v in vals.values()) or not finite(reference.get('auc')):
        return GateDecision('fairness',False,'reference','missing_group_or_label_support',vals)
    passed=(vals['dp_gap']<=max_dp and vals['eo_gap']<=max_eo and
            vals['auc']>=reference['auc']-max_auc_drop)
    return GateDecision('fairness',passed,'candidate' if passed else 'reference',
                        'fairness_utility_constraints_met' if passed else 'fairness_or_utility_violation',vals)

def prediction_stability(model, data, *, trials=5, rate=.05, seed=0):
    """Prediction stability measured only on validation examples (not test)."""
    idx=data['val'];x,e=data['x'],data['edges']
    if e is None or e.numel()==0:
        return {'preservation':None, 'probability_drift':None}
    with torch.no_grad():
        orig=torch.sigmoid(model(x,e)[idx]);out=[];drift=[]
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            for _ in range(trials):
                ep=perturb_edges(e,x.size(0),rate)
                p=torch.sigmoid(model(x,ep)[idx]);out.append(((p>=.5)==(orig>=.5)).float().mean().item());drift.append((p-orig).abs().mean().item())
    return {'preservation':sum(out)/len(out), 'probability_drift':sum(drift)/len(drift)}

def stability_gate(candidate, reference, *, min_preservation=.95, max_drift=.04,
                   max_extra_drift=.01):
    if not all(finite(candidate.get(k)) for k in ('preservation','probability_drift')):
        return GateDecision('robustness',False,'reference','stability_unavailable',candidate)
    extra=(candidate['probability_drift']-reference['probability_drift']) if finite(reference.get('probability_drift')) else 0.0
    passed=(candidate['preservation']>=min_preservation and candidate['probability_drift']<=max_drift and extra<=max_extra_drift)
    return GateDecision('robustness',passed,'candidate' if passed else 'reference',
                        'stability_constraints_met' if passed else 'unstable_under_edge_edits',candidate)

def explanation_gate(report, *, min_std=1e-4, required_random_margin=.005):
    scores=report.get('scores',{}); learned=report.get('learned',{});random=report.get('random',{})
    std=scores.get('std'); suf_l=learned.get('sufficiency_mae');suf_r=random.get('sufficiency_mae')
    margin=suf_r-suf_l if finite(suf_l) and finite(suf_r) else None
    ok=(finite(std) and std>=min_std and report.get('order_invariant',False) and
        finite(margin) and margin>=required_random_margin)
    return GateDecision('explanation',ok,'learned_mask' if ok else 'abstain',
                        'basic_mask_screen_passed_not_proof' if ok else 'mask_integrity_or_random_control_failure',
                        {'score_std':std,'random_minus_learned_sufficiency_mae':margin})

def retention_gate(before,after, *, max_auc_forgetting=.03):
    a=before.get('auc');b=after.get('auc')
    forgetting=a-b if finite(a) and finite(b) else None
    ok=finite(forgetting) and forgetting<=max_auc_forgetting
    return GateDecision('past_learning',ok,'updated_model' if ok else 'previous_checkpoint',
                        'retention_within_tolerance' if ok else 'forgotten_or_unmeasurable',
                        {'old_task_auc_forgetting':forgetting})

def evaluate_module_stack(base, candidate, data, *, reference=None, seed=0,
                          max_dp=.10,max_eo=.10,max_auc_drop=.03,
                          min_preservation=.95,max_drift=.04):
    """Screen a candidate jointly for fairness, prediction utility, perturbation stability.
    If any gate fails, return reference checkpoint; do not claim it meets failed constraints.
    """
    ref=reference or base; idx=data['val'];e=data['edges'];x=data['x']
    with torch.no_grad():
        r=metric(ref(x,e),data['y'],data['s'],idx)
        c=metric(candidate(x,e),data['y'],data['s'],idx)
    fg=fairness_gate(c,r,max_dp=max_dp,max_eo=max_eo,max_auc_drop=max_auc_drop)
    sr=prediction_stability(ref,data,seed=seed)
    sc=prediction_stability(candidate,data,seed=seed)
    rg=stability_gate(sc,sr,min_preservation=min_preservation,max_drift=max_drift)
    selected=candidate if fg.passed and rg.passed else ref
    return selected, {'fairness':fg.report(),'robustness':rg.report(),
                       'selected':'candidate' if selected is candidate else 'reference_checkpoint',
                       'reference_metrics':r,'candidate_metrics':c,
                       'note':'Fallback checkpoint is not certified fair or robust; report all failed gates.'}
