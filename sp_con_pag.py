import torch
import torch.nn as nn
import torch.nn.functional as F


class RAFFusion(nn.Module):
    """
    Relation-aware Adaptive Fusion (论文级版本)
    用于融合 ID embedding 和 multimodal embedding
    """

    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # channel projection（类似论文里的 channel descriptor）
        self.proj_id = nn.Linear(dim, dim)
        self.proj_modal = nn.Linear(dim, dim)

        # relation modeling（核心）
        self.relation_mlp = nn.Sequential(
            nn.Linear(num_heads * num_heads, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

        # residual gates
        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))

        # final normalization（很重要，提升稳定性）
        self.norm = nn.LayerNorm(dim)

    def forward(self, id_emb, modal_emb):
        # 支持 (B, L, D) 或 (B, D)
        orig_shape = id_emb.shape
        id_emb_flat = id_emb.view(-1, self.dim)
        modal_emb_flat = modal_emb.view(-1, self.dim)
        
        B_L = id_emb_flat.size(0)

        # 1. 投影
        id_proj = self.proj_id(id_emb_flat)
        modal_proj = self.proj_modal(modal_emb_flat)

        # 2. 多头关系建模 (核心修复：使用 chunk 代替强制 view)
        # 无论 dim 是多少，chunk 都能把它切成 num_heads 份
        # id_split shape: (num_heads, B_L, dim/num_heads)
        id_split = torch.stack(torch.chunk(id_proj, self.num_heads, dim=-1), dim=0)
        modal_split = torch.stack(torch.chunk(modal_proj, self.num_heads, dim=-1), dim=0)

        # 调整维度顺序以便 bmm: (num_heads, B_L, head_dim)
        # 我们直接在 head 维度上做 affinity
        # 这里的逻辑是计算 head 之间的相关性矩阵
        # 换一种更稳健的写法：(B_L, heads, head_dim)
        id_split = id_split.transpose(0, 1) 
        modal_split = modal_split.transpose(0, 1)

        # 这里的转置要小心，确保计算的是 (heads, heads) 的关联
        relation = torch.bmm(id_split, modal_split.transpose(1, 2)) # (B_L, heads, heads)
        
        # 归一化一下，防止数值过大
        relation = relation / (id_split.size(-1) ** 0.5)
        relation = relation.view(B_L, -1) # (B_L, num_heads * num_heads)

        relation_feat = self.relation_mlp(relation) # (B_L, D)

        # 3. 自适应门控
        gate_id = torch.sigmoid(id_proj + self.alpha * relation_feat)
        gate_modal = torch.sigmoid(modal_proj + self.beta * relation_feat)

        # 4. 融合与归一化
        fused = gate_id * id_emb_flat + gate_modal * modal_emb_flat
        output = self.norm(fused)

        return output.view(orig_shape)