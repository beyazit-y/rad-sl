import torch
import torch.nn as nn

class ContrastiveLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, l_dfa_embeddings, r_dfa_embeddings):
        max_dist = -float('inf')
        for l_state in l_dfa_embeddings:
            for r_state in r_dfa_embeddings:
                curr_dist = torch.norm(l_state - r_state, p=2, dim=-1)
                max_dist = max(max_dist, curr_dist)
        return max_dist