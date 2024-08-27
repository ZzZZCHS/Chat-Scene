# import torch
# import os


# for split in ["train", "val"]:
#     attrs = torch.load(f"annotations/scannet_{split}_attributes.pt", map_location='cpu')
#     for scan_id in attrs.keys():
#         locs = attrs[scan_id]['locs']
#         locs = [locs[i:i+6] for i in range(0, locs.shape[0], 6)]
#         locs = torch.stack(locs, dim=0)
#         attrs[scan_id]['locs'] = locs
#     torch.save(attrs, f"annotations/scannet_{split}_attributes.pt")

# import re

# def replace_substrings(text, replace_map):
#     pattern = re.compile('|'.join(re.escape(key) for key in replace_map.keys()))
#     def replacer(match):
#         return replace_map[match.group(0)]
#     return pattern.sub(replacer, text)

# # Example usage
# replace_map = {
#     "<OBJ001>": "replaced_string_1",
#     "<OBJ002>": "replaced_string_2",
# }

# text = "<OBJ001>This is a sample text with <OBJ001> and <OBJ002>.<OBJ001>"
# new_text = replace_substrings(text, replace_map)

# print(new_text)


import torch
import torch.nn as nn
from einops import rearrange

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim, heads, dim_head, dropout):
        inner_dim = heads * dim_head
        self.scale = dim_head ** -0.5
        self.heads = heads
        
        self.to_q = nn.Linear(query_dim, inner_dim)
        self.to_k = nn.Linear(context_dim, inner_dim)
        self.to_v = nn.Linear(context_dim, inner_dim)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(p=dropout)
        )
    
    def forward(self, query, context, mask):
        q = self.to_q(query)
        k = self.to_k(context)
        v = self.to_v(context)
        
        q, k, v = map(lambda t: rearrange(t, "b l (h d) -> (b h) l d", h=self.heads), (q, k, v))
        
        sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale
        
        # mask shape (b, i, j)
        if mask is not None:
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = rearrange(mask, "b i j -> (b h) i j", h=self.heads)
            sim.masked_fill_(~mask, max_neg_value)
        
        attn = sim.softmax(dim=-1)
        
        out = torch.einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) l d -> b l (h d)", h=self.heads)
        return self.to_out(out)
        