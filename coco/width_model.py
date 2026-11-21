import math
import torch
from torch import nn
from torch.nn import functional as F
SIDE=32
PATCH=8
WIDTH=192
DEPTH=2
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

def sinusoidal_1d(pos, dim):
    half = dim // 2
    omega = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32) / max(half, 1))
    args = pos[:, None] * omega[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


def pos_embed_2d(grid, dim):
    y, x = torch.meshgrid(torch.arange(grid, dtype=torch.float32),
                          torch.arange(grid, dtype=torch.float32), indexing="ij")
    return torch.cat([sinusoidal_1d(y.flatten(), dim // 2),
                      sinusoidal_1d(x.flatten(), dim // 2)], dim=-1)[None]


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps) * self.weight


def apply_axis_rope(x, coords):
    # x: (B, heads, tokens, axis_dim), axis_dim is even.
    d = x.shape[-1]
    inv = torch.exp(-math.log(10000) * torch.arange(0, d, 2, device=x.device, dtype=x.dtype) / d)
    angle = coords.to(device=x.device, dtype=x.dtype)[:, None] * inv[None]
    cos, sin = angle.cos()[None, None], angle.sin()[None, None]
    pairs = x.reshape(*x.shape[:-1], d // 2, 2)
    a, b = pairs.unbind(-1)
    return torch.stack([a * cos - b * sin, a * sin + b * cos], dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, dim, heads, grid):
        super().__init__()
        self.heads, self.head_dim, self.grid = heads, dim // heads, grid
        assert self.head_dim % 4 == 0
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.qnorm = RMSNorm(self.head_dim)
        self.knorm = RMSNorm(self.head_dim)
        self.proj = nn.Linear(dim, dim)
        yy, xx = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij")
        self.register_buffer("coord_y", yy.flatten().float(), persistent=False)
        self.register_buffer("coord_x", xx.flatten().float(), persistent=False)

    def forward(self, x, num_prefix=0):
        b, n, d = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.qnorm(q), self.knorm(k)
        half = self.head_dim // 2
        cy, cx = self.coord_y, self.coord_x
        if num_prefix:
            zeros = torch.zeros(num_prefix, device=cy.device)
            cy, cx = torch.cat([zeros, cy]), torch.cat([zeros, cx])
        q = torch.cat([apply_axis_rope(q[..., :half], cy),
                       apply_axis_rope(q[..., half:], cx)], dim=-1)
        k = torch.cat([apply_axis_rope(k[..., :half], cy),
                       apply_axis_rope(k[..., half:], cx)], dim=-1)
        a = torch.softmax((q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim), dim=-1)
        return self.proj((a @ v).transpose(1, 2).reshape(b, n, d))


class SwiGLU(nn.Module):
    def __init__(self, dim, ratio=4):
        super().__init__()
        hidden = int(dim * ratio * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden)
        self.w3 = nn.Linear(hidden, dim)

    def forward(self, x):
        a, b = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(a) * b)


def modulate(x, shift, scale):
    return x * (1 + scale[:, None]) + shift[:, None]


class JiTLikeBlock(nn.Module):
    def __init__(self, dim, heads, grid):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(dim), RMSNorm(dim)
        self.attn = Attention(dim, heads, grid)
        self.mlp = SwiGLU(dim)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x, c, num_prefix=0):
        s1, a1, g1, s2, a2, g2 = self.ada(c).chunk(6, dim=-1)
        x = x + g1[:, None] * self.attn(modulate(self.norm1(x), s1, a1), num_prefix)
        x = x + g2[:, None] * self.mlp(modulate(self.norm2(x), s2, a2))
        return x


class TinyJiT(nn.Module):
    """Small JiT/ViT-style unconditional denoiser for a local mechanism test."""
    def __init__(self, width, depth=4, patch=PATCH):
        super().__init__()
        self.patch, self.grid, self.width = patch, SIDE // patch, width
        pca_dim = width // 4
        self.patch1 = nn.Conv2d(3, pca_dim, patch, stride=patch, bias=False)
        self.patch2 = nn.Conv2d(pca_dim, width, 1)
        self.register_buffer("pos", pos_embed_2d(self.grid, width), persistent=False)
        self.time_mlp = nn.Sequential(nn.Linear(64, width), nn.SiLU(), nn.Linear(width, width))
        self.null_label = nn.Parameter(torch.empty(1, width))
        self.in_context_len = 8
        self.in_context_start = depth // 3
        self.in_context_posemb = nn.Parameter(torch.empty(1, self.in_context_len, width))
        nn.init.normal_(self.null_label, std=.02)
        nn.init.normal_(self.in_context_posemb, std=.02)
        self.blocks = nn.ModuleList([JiTLikeBlock(width, width // 32, self.grid) for _ in range(depth)])
        self.norm = RMSNorm(width)
        self.final_ada = nn.Sequential(nn.SiLU(), nn.Linear(width, 2 * width))
        self.out = nn.Linear(width, patch * patch * 3)
        self.apply(self._basic_init)
        nn.init.xavier_uniform_(self.patch1.weight.view(self.patch1.weight.shape[0], -1))
        nn.init.xavier_uniform_(self.patch2.weight.view(self.patch2.weight.shape[0], -1))
        nn.init.zeros_(self.patch2.bias)
        nn.init.normal_(self.time_mlp[0].weight, std=.02)
        nn.init.normal_(self.time_mlp[2].weight, std=.02)
        for block in self.blocks:
            nn.init.zeros_(block.ada[-1].weight)
            nn.init.zeros_(block.ada[-1].bias)
        nn.init.zeros_(self.final_ada[-1].weight); nn.init.zeros_(self.final_ada[-1].bias)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    @staticmethod
    def _basic_init(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def time_embed(self, t):
        half = 32
        freq = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        phase = t.float()[:, None] * freq[None]
        return self.time_mlp(torch.cat([torch.cos(phase), torch.sin(phase)], -1))

    def forward(self, image, t):
        b = len(image)
        c = self.time_embed(t) + self.null_label
        x = self.patch2(self.patch1(image)).flatten(2).transpose(1, 2) + self.pos
        prefix = 0
        for i, block in enumerate(self.blocks):
            if i == self.in_context_start:
                context = c[:, None].expand(-1, self.in_context_len, -1) + self.in_context_posemb
                x = torch.cat([context, x], dim=1)
                prefix = self.in_context_len
            x = block(x, c, prefix)
        if prefix:
            x = x[:, prefix:]
        shift, scale = self.final_ada(c).chunk(2, -1)
        x = self.out(modulate(self.norm(x), shift, scale))
        x = x.reshape(b, self.grid, self.grid, self.patch, self.patch, 3)
        return torch.einsum("nhwpqc->nchpwq", x).reshape(b, 3, SIDE, SIDE)


def model(b,seed):
 torch.manual_seed(8100+seed)
 m=TinyJiT(WIDTH,depth=DEPTH,patch=8)
 # All backbone/output tensors start identically across bottleneck sizes.
 torch.manual_seed(9300+seed)
 m.patch1=nn.Conv2d(3,b,8,stride=8,bias=False)
 m.patch2=nn.Conv2d(b,WIDTH,1,bias=True)
 nn.init.orthogonal_(m.patch1.weight.reshape(b,-1))
 nn.init.orthogonal_(m.patch2.weight.reshape(WIDTH,b));nn.init.zeros_(m.patch2.bias)
 return m.to(DEVICE)
