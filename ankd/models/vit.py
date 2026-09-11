"""Vision transformer sized for 32x32 CIFAR images.

Torchvision's vit_b_16 is a 224x224 ImageNet design and is unusable at this
resolution, so the architecture is defined here: 4x4 patches over 32x32 (64
tokens + CLS), learned position embeddings, pre-norm blocks.

ViT-8 and ViT-4 differ only in depth -- 8 blocks against 4, with patch size,
width, head count and MLP ratio held fixed, so the student is the teacher with
half the blocks. Depth is the usual compression axis for ViT distillation;
halving the patch size instead would quadruple the token count and make the
student larger than the teacher.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """Multi-head self-attention; pre-norm is applied by the enclosing Block."""

    def __init__(self, dim, heads, drop=0.0):
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = drop
        self.proj_drop = nn.Dropout(drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_drop if self.training else 0.0)
        return self.proj_drop(self.proj(x.transpose(1, 2).reshape(B, N, C)))


class Block(nn.Module):
    """x + attn(norm(x)), then x + mlp(norm(x))."""

    def __init__(self, dim, heads, mlp_ratio=2.0, drop=0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.norm1, self.attn = nn.LayerNorm(dim), Attention(dim, heads, drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, dim), nn.Dropout(drop),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class ViT(nn.Module):
    def __init__(self, depth, num_classes=10, img_size=32, patch=4, dim=256,
                 heads=4, mlp_ratio=2.0, drop=0.1):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)
        n_patches = (img_size // patch) ** 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, dim))
        self.pos_drop = nn.Dropout(drop)

        self.blocks = nn.ModuleList([Block(dim, heads, mlp_ratio, drop) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)          # (B, N, dim)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = self.pos_drop(torch.cat((cls, x), dim=1) + self.pos_embed)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x)[:, 0])                        # CLS token


def vit8(num_classes=10, **kw):
    return ViT(8, num_classes, **kw)


def vit4(num_classes=10, **kw):
    return ViT(4, num_classes, **kw)
