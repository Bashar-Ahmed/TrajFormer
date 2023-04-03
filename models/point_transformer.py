import torch
from torch import nn, einsum
from einops import repeat

from models.sinkhorn import sinkhorn

def batched_index_select(values, indices, dim = 1):
    value_dims = values.shape[(dim + 1):]
    values_shape, indices_shape = map(lambda t: list(t.shape), (values, indices))
    indices = indices[(..., *((None,) * len(value_dims)))]
    indices = indices.expand(*((-1,) * len(indices_shape)), *value_dims)
    value_expand_len = len(indices_shape) - (dim + 1)
    values = values[(*((slice(None),) * dim), *((None,) * value_expand_len), ...)]

    value_expand_shape = [-1] * len(values.shape)
    expand_slice = slice(dim, (dim + value_expand_len))
    value_expand_shape[expand_slice] = indices.shape[expand_slice]
    values = values.expand(*value_expand_shape)

    dim += value_expand_len
    return values.gather(dim, indices)

class PointTransformerLayer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        attr,
        pos_mlp_hidden_dim = 128,
        attn_mlp_hidden_mult = 4,
        num_neighbors = None
    ):
        super().__init__()
        
        self.num_neighbors = num_neighbors
        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)
        self.pos_mlp = nn.Sequential(
            nn.Linear(attr, pos_mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(pos_mlp_hidden_dim, dim)
        )
        # cuda1 = torch.device('cuda:0')
        # self.pos_mlp.to(cuda1)
        self.attn_mlp = nn.Sequential(
            nn.Linear(dim, dim * attn_mlp_hidden_mult),
            nn.ReLU(),
            nn.Linear(dim * attn_mlp_hidden_mult, dim),
        )

    def forward(self, x, pos, mask = None):

        n, num_neighbors = x.shape[1], self.num_neighbors
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        rel_pos = pos[:, :, None, :] - pos[:, None, :, :]
        rel_pos_emb = self.pos_mlp(rel_pos)
        # cuda1 = torch.device('cuda:0')
        # q.to(cuda1)
        # k.to(cuda1)
        qk_rel = q[:, :, None, :] - k[:, None, :, :]

        if mask is not None:
            mask = mask[:, :, None] * mask[:, None, :]

        v = repeat(v, 'b j d -> b i j d', i = n)

        if num_neighbors is not None and num_neighbors < n:
            rel_dist = rel_pos.norm(dim = -1)
            if mask is not None:
                mask_value = torch.finfo(rel_dist.dtype).max
                rel_dist.masked_fill_(~mask, mask_value)

            dist, indices = rel_dist.topk(num_neighbors, largest = False)
            v = batched_index_select(v, indices, dim = 2)
            qk_rel = batched_index_select(qk_rel, indices, dim = 2)
            rel_pos_emb = batched_index_select(rel_pos_emb, indices, dim = 2)
            mask = batched_index_select(mask, indices, dim = 2) if mask is not None else None

        v = v + rel_pos_emb
        sim = self.attn_mlp(qk_rel + rel_pos_emb)

        if mask is not None:
            mask_value = -torch.finfo(sim.dtype).max
            sim.masked_fill_(~mask[..., None], mask_value)

        attn = sinkhorn(sim)
        agg = einsum('b i j d, b i j d -> b i d', attn, v)
        return agg
