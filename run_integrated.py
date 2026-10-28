"""End-to-end research scaffold: graph bridge + joint fairness/stability gates + mask abstention + retention gate.
Synthetic smoke only; no real-dataset performance claims.
"""
import argparse, json, os
from frme.core import synthetic_graph,seed_all,metric
from frme.graph_bridge import bridge_graph,validation_fallback
from frme.train import train_predictor, train_explainer, evaluate_explainer
from frme.controller import evaluate_module_stack, explanation_gate, retention_gate
from frme.federated import freeze_teacher,replay_memory

p=argparse.ArgumentParser()
p.add_argument('--epochs',type=int,default=35)
p.add_argument('--seeds',type=int,default=1)
p.add_argument('--out',default='results/integrated.json')
p.add_argument('--budget',type=float,default=.5)
p.add_argument('--max-dp',type=float,default=.10)
p.add_argument('--max-eo',type=float,default=.10)
p.add_argument('--max-auc-drop',type=float,default=.03)
p.add_argument('--min-preservation',type=float,default=.90)
p.add_argument('--max-drift',type=float,default=.10)
a=p.parse_args()
os.makedirs(os.path.dirname(a.out) or '.',exist_ok=True)
results=[]
for seed in range(a.seeds):
    seed_all(seed);d=synthetic_graph(seed=seed)
    bridge=bridge_graph(d['x'],d['edges'])
    if bridge['edges'] is not None:d['edges']=bridge['edges']
    feature=train_predictor(d,epochs=a.epochs,seed=seed,model_type='mlp')
    selected=feature;route='features';joint=None;explanation={'status':'abstain_no_graph'}
    if bridge['edges'] is not None:
        graph=train_predictor(d,epochs=a.epochs,seed=seed,model_type='gcn')
        gm=metric(graph(d['x'],d['edges']),d['y'],d['s'],d['val'])
        fm=metric(feature(d['x']),d['y'],d['s'],d['val'])
        route,_=validation_fallback(gm['auc'],fm['auc'])
        if route=='graph':
            # Baseline is a reference checkpoint, not a certified fair/robust model.
            candidate=train_predictor(d,epochs=a.epochs,seed=seed,model_type='gcn',
                                      init_model=graph,lam_fair=.15,lr=.005)
            selected,joint=evaluate_module_stack(graph,candidate,d,seed=seed,
                       max_dp=a.max_dp,max_eo=a.max_eo,max_auc_drop=a.max_auc_drop,
                       min_preservation=a.min_preservation,max_drift=a.max_drift)
            ex=train_explainer(selected,d,epochs=a.epochs,seed=seed,budget=a.budget,lam_rob=.2)
            er=evaluate_explainer(selected,ex,d,budget=a.budget,seed=seed)
            eg=explanation_gate(er)
            explanation={'status':eg.route,'gate':eg.report(),'diagnostics':er}
    old=freeze_teacher(selected)
    new=synthetic_graph(seed=seed+900)
    updated=train_predictor(new,epochs=a.epochs,seed=seed,init_model=old,
                            teacher=old,replay=replay_memory(d,seed=seed),
                            lam_fair=.05,lam_replay=.4,lam_kd=.1,lam_prox=.001,prev=old)
    old_before=metric(selected(d['x'],d['edges']),d['y'],d['s'],d['val'])
    old_after=metric(updated(d['x'],d['edges']),d['y'],d['s'],d['val'])
    retain=retention_gate(old_before,old_after)
    active=updated if retain.passed else old
    with __import__('torch').no_grad():
        test_metrics=metric(active(d['x'],d['edges']),d['y'],d['s'],d['test'])
    results.append({'seed':seed,'bridge':{'route':bridge['route'],'attempts':bridge['attempts'],
                    'graph_utility_route':route},'joint_validation':joint,
                    'explanation':explanation,'retention':retain.report(),'test_metrics':test_metrics,
                    'warning':'Synthetic demonstration; no method outperformance established.'})
with open(a.out,'w') as f:json.dump(results,f,indent=2)
print('Saved:',a.out)
for row in results:
    print('seed',row['seed'],'graph',row['bridge']['graph_utility_route'],
          'joint',row['joint_validation']['selected'] if row['joint_validation'] else 'N/A',
          'mask',row['explanation']['status'],'retention',row['retention']['route'])
