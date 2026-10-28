"""Federated-learning-inspired utilities. Simulated FedAvg, not secure FL."""
import copy
import torch
from .train import train_predictor


def fedavg_round(global_model,clients,epochs=10,seed=0,lam_prox=.001):
    states=[];sizes=[]
    for c,data in enumerate(clients):
        local=train_predictor(data,epochs=epochs,seed=seed+c,prev=global_model,lam_prox=lam_prox,init_model=global_model)
        states.append(local.state_dict());sizes.append(len(data['train']))
    total=sum(sizes);new=copy.deepcopy(global_model)
    with torch.no_grad():
        for k in new.state_dict():
            new.state_dict()[k].copy_(sum(s[k]*(n/total) for s,n in zip(states,sizes)))
    return new


def freeze_teacher(model):
    t=copy.deepcopy(model).eval()
    for p in t.parameters():p.requires_grad_(False)
    return t


def replay_memory(data,cap=48,seed=0):
    g=torch.Generator().manual_seed(seed)
    idx=data['train'][torch.randperm(len(data['train']),generator=g)[:cap]]
    return {k:data[k].clone() for k in ('x','edges','y')}|{'idx':idx.clone()}
