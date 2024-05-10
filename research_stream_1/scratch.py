# %%
from transformer_lens import HookedTransformer, HookedSAETransformer
from transformer_lens import utils as tutils
import torch
import numpy as np
import random


SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

torch.set_grad_enabled(False)


model = HookedTransformer.from_pretrained("gpt2-small")

n_layers = model.cfg.n_layers
d_model = model.cfg.d_model
n_heads = model.cfg.n_heads
d_head = model.cfg.d_head
d_mlp = model.cfg.d_mlp
d_vocab = model.cfg.d_vocab

# %%
prompt = \
"""
15. Dog
16. Cat
17. Hamster
"""
answer = "18"

tutils.test_prompt(prompt, answer, model, prepend_space_to_answer=False)


# %%
