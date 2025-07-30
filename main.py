import dfa
import dfa_samplers

# 0. Let rad_sampler be a RAD DFA distribution
rad_sampler = dfa_samplers.RADSampler(n_tokens=10)

# 1. Sample two distinct DFAs from rad_sampler, call dfa_l and dfa_r.
dfa_l = rad_sampler.sample()
dfa_r = rad_sampler.sample()

# 2. Compute a minimum distinguishing string, w, dfa_l and dfa_r, e.g., using a shortestpath algorithm.
# TODO
# Hints
# See https://github.com/mvcisback/dfa for helper functions and operations defined on DFAs.

# 3. Compute the embeddings for dfa_l and dfa_r, i.e., encoder(dfa_l) and encoder(dfa_r).
# TODO
# Hints
# See https://github.com/RAD-Embeddings/rad-embeddings/blob/main/rad_embeddings/model.py for model used for the encoder.
# See _obs2feat starting from line 24 in https://github.com/RAD-Embeddings/rad-embeddings/blob/main/rad_embeddings/utils/utils.py#L34 to see how to pass DFAs to the encoder.

# 4. Compute the pairs of states visited by w and their corresonding embeddings.
# TODO
# Hints
# From the encoder model instead of returning the embedding of the current state (as if https://github.com/RAD-Embeddings/rad-embeddings/blob/main/rad_embeddings/model.py#L44), return the embeddings for each visited state.

# 5. To compute loss(dfa_l, dfa_r), calculate their relative pair-wise l2-distance and return the maximum.
# TODO
# Hints
# d = torch.norm(feat1 - feat2, p=2, dim=-1)

# 6. Finally, compute the gradient of loss(dfa_l, dfa_r), to train encoder, e.g., using Adam.
# TODO
# Hints
# Use pytorch for this

# 7. Turn this procedure into a training loop
# TODO
# Hints
# Sample batches of such problems and this in a loop.