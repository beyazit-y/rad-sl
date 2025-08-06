import torch
import numpy as np
from dfa.utils import dfa2dict
from collections import OrderedDict
from torch_geometric.data import Data, Batch

feature_inds = {"temp": -5, "rejecting": -4, "accepting": -3, "init": -2, "normal": -1}

def dfa2feat(dfa, n_tokens):
    data = Batch.from_data_list([_dfa2feat(dfa, n_tokens=n_tokens)])
    max_i = data.n_nodes.max().item()
    node_mask = torch.tensor([data.n_nodes[i] for i in range(data.batch_size) for _ in range(data.n_nodes[i])])
    edge_mask = torch.tensor([data.n_nodes[i] for i in range(data.batch_size) for _ in range(data.n_edges[i])])
    data.active_node_indices = torch.stack([i < node_mask for i in range(max_i)])
    data.active_edge_indices = torch.stack([i < edge_mask for i in range(max_i)])
    return data

def _dfa2feat(dfa, n_tokens):
    feature_size = n_tokens + len(feature_inds)
    dfa_dict, s_init = dfa2dict(dfa)
    nodes = OrderedDict({s: np.zeros(feature_size) for s in dfa_dict.keys()})
    if len(nodes) == 1:
        edges = [(0, 0)]
    else:
        edges = [(s, s) for s in nodes]
    for s in dfa_dict.keys():
        label, transitions = dfa_dict[s]
        leaving_transitions = [1 if s != transitions[a] else 0 for a in transitions.keys()]
        if s not in nodes:
            nodes[s] = np.zeros(feature_size)
        nodes[s][feature_inds["normal"]] = 1
        if s == s_init:
            nodes[s][feature_inds["init"]] = 1
        if label: # is accepting?
            nodes[s][feature_inds["accepting"]] = 1
        elif sum(leaving_transitions) == 0: # is rejecting?
            nodes[s][feature_inds["rejecting"]] = 1
        for e in dfa_dict.keys():
            if s == e:
                continue
            for a in transitions:
                if transitions[a] == e:
                    if (s, e) not in nodes:
                        nodes[(s, e)] = np.zeros(feature_size)
                        nodes[(s, e)][feature_inds["temp"]] = 1
                    nodes[(s, e)][a] = 1
                    s_idx = list(nodes.keys()).index(s)
                    t_idx = list(nodes.keys()).index((s, e))
                    e_idx = list(nodes.keys()).index(e)
                    # Reverse
                    if (e_idx, t_idx) not in edges:
                        edges.append((e_idx, t_idx))
                    if (t_idx, t_idx) not in edges:
                        edges.append((t_idx, t_idx))
                    if (t_idx, s_idx) not in edges:
                        edges.append((t_idx, s_idx))
    feat = torch.from_numpy(np.array(list(nodes.values())))
    edge_index = torch.from_numpy(np.array(edges)).T
    return Data(feat=feat, edge_index=edge_index, n_nodes=len(nodes), n_edges=len(edges))