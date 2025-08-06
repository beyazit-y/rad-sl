import torch
import torch.optim as optim
import dfa_samplers
from utils import feature_inds, dfa2feat
from min_string import compute_min_dist_string
from loss import ContrastiveLoss
from encoder import Encoder
import matplotlib.pyplot as plt

n_tokens = 10
num_steps = 100_000
learning_rate = 1e-5
rad_sampler = dfa_samplers.RADSampler(n_tokens)
in_feat_size = n_tokens + len(feature_inds)
dfa_embed_size = 32
model = Encoder(input_dim=in_feat_size, output_dim=dfa_embed_size)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = ContrastiveLoss()

# begin training loop here
print(f'Starting {num_steps} steps of training...')
model.train()
training_losses = []
for step in range(num_steps):

    dfa_l = rad_sampler.sample()
    dfa_r = rad_sampler.sample()
    w = compute_min_dist_string(dfa_l, dfa_r)
    # print(f'Minimum Distinguishing String: {"".join(str(c) for c in w)}')

    l_visited_states = torch.tensor(list(dfa_l.trace(w)))
    r_visited_states = torch.tensor(list(dfa_r.trace(w)))
    dfa_l_embeddings = model(dfa2feat(dfa_l, n_tokens), l_visited_states)
    dfa_r_embeddings = model(dfa2feat(dfa_r, n_tokens), r_visited_states)

    optimizer.zero_grad()
    loss = criterion(dfa_l_embeddings, dfa_r_embeddings)
    loss.backward()
    optimizer.step()

    training_losses.append(loss.item())
    if (step + 1) % 1000 == 0:
        print(f'Finished training step {step + 1} with loss: {loss.item()}')

plt.plot([step for step in range(num_steps)], training_losses)
plt.xlabel("Training Step")
plt.ylabel("Loss")
plt.yscale("log")
plt.title("RL-Free Contrastive Loss of DFA Encoder")
plt.grid(True)
plt.show()