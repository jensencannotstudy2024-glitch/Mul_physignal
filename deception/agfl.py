"""AGFL 图滤波层（照 `04_AGFL_Mathematics.docx` 实现）。

一个 `AGFLLayer` 在给定邻接掩码的图上做 K 跳传播：
  H  = LN(H_enc + b)
  S  = (H Wq)(H Wk)^T / sqrt(a)
  A  = 每个节点在合法邻居中保留 top-kappa 后做 softmax（行和为 1，不必对称）
  V0 = H Wv,  U_l = A V_{l-1},  V_l = U_l * ||V0|| / (||U_l|| + eps)
  T  = sum_l alpha_l V_l,  alpha = softmax(c)
  out = LN(H + T Wo)

K = 0 时完全不构造邻接，只用 V0 —— 这是"层存在但无图交换"的消融条件。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_edge_mask(n_nodes: int, group_sizes: list[int] | None = None) -> torch.Tensor:
    """合法有向边：mask[i, j] 为 True 表示 i 可以接收 j 的消息。

    自环一律排除。给定 group_sizes 时返回块对角掩码（模态内图，无跨模态边）。
    """
    if group_sizes is None:
        mask = torch.ones(n_nodes, n_nodes, dtype=torch.bool)
        mask.fill_diagonal_(False)
        return mask
    if sum(group_sizes) != n_nodes:
        raise ValueError("group_sizes must sum to n_nodes.")
    mask = torch.zeros(n_nodes, n_nodes, dtype=torch.bool)
    start = 0
    for size in group_sizes:
        block = torch.ones(size, size, dtype=torch.bool)
        block.fill_diagonal_(False)
        mask[start:start + size, start:start + size] = block
        start += size
    return mask


def masked_topk_softmax(scores: torch.Tensor, mask: torch.Tensor, kappa: int) -> torch.Tensor:
    """先在掩码上取 top-kappa，再只在选中集合内归一化。"""
    if scores.dim() != 3 or scores.shape[-1] != scores.shape[-2]:
        raise ValueError("scores must have shape (batch, n, n).")
    mask = mask.to(device=scores.device, dtype=torch.bool)
    if tuple(mask.shape) != tuple(scores.shape[-2:]):
        raise ValueError("mask shape must match the last two score dimensions.")
    legal = mask.sum(dim=-1)
    if int(legal.min()) == 0:
        row = int(torch.argmin(legal))
        raise ValueError(f"Node {row} has no eligible neighbour; refusing to emit an all-zero row.")
    k = min(int(kappa), int(legal.min()))
    if k < 1:
        raise ValueError("kappa must be at least 1.")
    fill = torch.finfo(scores.dtype).min
    masked = scores.masked_fill(~mask.unsqueeze(0), fill)
    values, indices = masked.topk(k, dim=-1)
    weights = torch.softmax(values, dim=-1)
    adjacency = torch.zeros_like(scores)
    adjacency.scatter_(-1, indices, weights)
    return adjacency


class AGFLLayer(nn.Module):
    """单层 AGFL：共享参数的图滤波。"""

    def __init__(self, d: int = 16, a: int = 16, hops: int = 2,
                 kappa: int = 5, eps: float = 1e-6):
        super().__init__()
        if hops < 0:
            raise ValueError("hops must be non-negative.")
        self.d, self.a, self.hops, self.kappa, self.eps = int(d), int(a), int(hops), int(kappa), float(eps)
        self.bias = nn.Parameter(torch.zeros(self.d))
        self.Wq = nn.Linear(self.d, self.a, bias=False)
        self.Wk = nn.Linear(self.d, self.a, bias=False)
        self.Wv = nn.Linear(self.d, self.d, bias=False)
        self.Wo = nn.Linear(self.d, self.d, bias=False)
        self.c = nn.Parameter(torch.zeros(self.hops + 1))

    def forward(self, h: torch.Tensor, mask: torch.Tensor, return_hops: bool = False):
        if h.dim() != 3 or h.shape[-1] != self.d:
            raise ValueError("h must have shape (batch, nodes, d).")
        h = F.layer_norm(h + self.bias, (self.d,))
        v0 = self.Wv(h)
        hop_values = [v0]
        if self.hops > 0:
            scores = torch.matmul(self.Wq(h), self.Wk(h).transpose(-1, -2)) / math.sqrt(self.a)
            adjacency = masked_topk_softmax(scores, mask, self.kappa)
            value = v0
            for _ in range(self.hops):
                mixed = torch.matmul(adjacency, value)
                norm0 = v0.norm(dim=-1, keepdim=True)
                norm_mixed = mixed.norm(dim=-1, keepdim=True)
                value = mixed * norm0 / (norm_mixed + self.eps)
                hop_values.append(value)
        stacked = torch.stack(hop_values, dim=0)
        weights = torch.softmax(self.c, dim=0).view(-1, 1, 1, 1)
        combined = (weights * stacked).sum(dim=0)
        out = F.layer_norm(h + self.Wo(combined), (self.d,))
        if return_hops:
            return out, hop_values
        return out


class FixedProjection(nn.Module):
    """把折内 sklearn PCA 基固定成 torch 线性投影。"""

    def __init__(self, mean: torch.Tensor, components: torch.Tensor):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("components", components)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) @ self.components.transpose(0, 1)


class AGFLModel(nn.Module):
    """编码器输出 H → [intra] → [inter] → PCA 基 → Linear。"""

    def __init__(self, config: str, sizes: list[int], d: int, a: int, hops: int,
                 kappa: int, pca_mean: torch.Tensor, pca_components: torch.Tensor):
        super().__init__()
        if config not in {"none", "k0", "intra", "inter", "combined"}:
            raise ValueError(f"Unknown AGFL config {config!r}.")
        self.config = config
        self.sizes = list(sizes)
        shared_hops = 0 if config == "k0" else hops
        self.intra = (AGFLLayer(d, a, shared_hops, kappa)
                      if config in {"intra", "combined"} else None)
        self.inter = (AGFLLayer(d, a, shared_hops, kappa)
                      if config in {"inter", "combined"} else None)
        self.k0 = AGFLLayer(d, a, 0, kappa) if config == "k0" else None
        self.projection = FixedProjection(pca_mean, pca_components)
        n_components = int(pca_components.shape[0])
        self.head = nn.Linear(n_components, 2)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if self.k0 is not None:
            h = self.k0(h, build_edge_mask(h.shape[1]))
        if self.intra is not None:
            blocks, start = [], 0
            for size in self.sizes:
                blocks.append(self.intra(h[:, start:start + size], build_edge_mask(size)))
                start += size
            h = torch.cat(blocks, dim=1)
        if self.inter is not None:
            h = self.inter(h, build_edge_mask(h.shape[1]))
        flat = h.flatten(1)
        return self.head(self.projection(flat))
