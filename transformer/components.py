import torch
import torch.nn.functional as F
import einops
import math


def gelu(x):
    sigmoid_x = (1 + torch.e**(-1.702 * x) )**(-1)
    return x * sigmoid_x


class MLP(torch.nn.Module):
    def __init__(self, d_in, d_mlp):
        super().__init__()
        self.W_in = torch.nn.Parameter(
            torch.randn(size=(d_in, d_mlp)) * torch.sqrt(torch.tensor(2.0 / d_in))
        )
        self.W_out = torch.nn.Parameter(
            torch.randn(size=(d_mlp, d_in)) * torch.sqrt(torch.tensor(2.0 / d_mlp))
        )
        self.b_in = torch.nn.Parameter(
            torch.zeros(size=(d_mlp,))
        )
        self.b_out = torch.nn.Parameter(
            torch.zeros(size=(d_in,))
        )

    
    def forward(self, x):
        x = einops.einsum(x, self.W_in, "... d_in, d_in d_out -> ... d_out")
        x = x + self.b_in
        x = gelu(x)
        x = einops.einsum(x, self.W_out, "... d_mlp, d_mlp d_out -> ... d_out")
        x = x + self.b_out
        return x
    


class Attention(torch.nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.W_q = torch.nn.Parameter(
            torch.randn(size=(d_model, d_model)) # TODO: scale init?
        )
        self.W_k = torch.nn.Parameter(
            torch.randn(size=(d_model, d_model))
        )
        self.W_v = torch.nn.Parameter(
            torch.randn(size=(d_model, d_model)) * math.sqrt(2.0/d_model)
        )
        self.W_o = torch.nn.Parameter(
            torch.randn(size=(d_model, d_model)) * math.sqrt(2.0/d_model)
        )
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = 1/math.sqrt(self.d_head)

    def forward(self, x):
        # x shape is [batch, pos, d_model]
        # scores should be a [batch, head, pos, pos] matrix
        q = einops.einsum(x, self.W_q, "... pos d_model, d_model d_qs -> ... pos d_qs")
        k = einops.einsum(x, self.W_k, "... pos d_model, d_model d_ks -> ... pos d_ks")
        v = einops.einsum(x, self.W_v, "... pos d_model, d_model d_vs -> ... pos d_vs")

        q = einops.rearrange(q, "... pos (h d_h) -> ... pos h d_h", h=self.n_heads)
        k = einops.rearrange(k, "... pos (h d_h) -> ... pos h d_h", h=self.n_heads)

        scores = einops.einsum(q, k, "... pos_q h d_h, ... pos_k h d_h  -> ... h pos_q pos_k")

        nq = scores.shape[-2]
        nk = scores.shape[-1]
        mask = torch.triu(torch.ones(nq, nk), diagonal=1) * (-10e8)
        scores = scores + mask
        scores = scores * self.scale
        scores = F.softmax(scores, dim=-1)

        out = einops.einsum(scores, v, "... h pos_q pos_k, ... pos_k d_vs -> ... pos_q d_vs")
        out = einops.einsum(out, self.W_o, "... pos da, da dm -> ... pos dm")
        return out


