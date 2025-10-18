'''
Modified from the myna repository https://github.com/ghost-signal/myna
'''

import torch
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange
from einops import rearrange, repeat

from models import RopeALiBiModelComponents


# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype=torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head),
                FeedForward(dim, mlp_dim)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class Myna(nn.Module):
    def __init__(self, *, image_size, patch_size, latent_space, d_model, depth, heads, mlp_dim, channels=3, dim_head=64,
                 additional_patch_size=None, hybrid_mode: bool = False, mask_ratio: float = 0.0, use_cls=False):
        super().__init__()

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        self.patch_height = patch_height
        self.patch_width = patch_width

        self.additional_patch_size = additional_patch_size
        if additional_patch_size:
            patch_height_b, patch_width_b = pair(additional_patch_size)
            patch_dim_b = channels * patch_height_b * patch_width_b

            self.to_patch_embedding_b, self.pos_embedding_b = self._make_embeddings(
                patch_height_b, patch_width_b, patch_dim_b, d_model, image_height, image_width
            )

        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding, self.pos_embedding = self._make_embeddings(
            patch_height, patch_width, patch_dim, d_model, image_height, image_width
        )

        #self.transformer = Transformer(d_model, depth, heads, dim_head, mlp_dim)
        self.transformer = RopeALiBiModelComponents.CustomTransformerEncoder(num_layers=depth,
                                                                                      d_model=d_model,
                                                                                      num_heads=heads,
                                                                                      dim_feedforward=mlp_dim,
                                                                                      dropout=0.1)

        self.to_latent = nn.Identity()

        self.cls_token = torch.nn.Parameter(torch.randn(1, 1, d_model)) if use_cls else None

        self.mask_ratio = mask_ratio

        self.linear_head = nn.Linear(d_model, latent_space)

    def forward(self, img):
        device = img.device

        B, H, W = img.shape()

        x = self.to_patch_embedding(img)

        if self.alibi:
            coordinates = self.get_patch_coordinates(H, W)
        else:
            x += self.pos_embedding.to(device, dtype=x.dtype)

        if self.mask_ratio > 0.0:
            unmasked = self.mask_inputs(x, self.mask_ratio, device)
            x = x.gather(1, unmasked)
            if self.alibi:
                coordinates = coordinates.gather(1, unmasked)

        B, P, F = x.shape

        if self.cls_token is not None:
            cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=B)
            x = torch.cat((cls_tokens, x), dim=1)

        if self.alibi:
            x = self.transformer(x, coords=coordinates)
        else:
            x = self.transformer(x)


        if self.cls_token is not None:
            x = x[:, 0]
        else:
            x = x.mean(dim=1)

        x = self.to_latent(x)
        return self.linear_head(x)

    def mask_inputs(self, x: torch.Tensor, mask_ratio: float, device: str):
        ''' Input masking for contrastive learning '''
        # input B, N, D --> output B, N * (1 - mask_ratio), D
        B, N, _ = x.shape
        n_masked = int(mask_ratio * N)
        indices = torch.stack([torch.randperm(N) for _ in range(B)])
        unmask_indices = indices[:, n_masked:].to(device)

        return unmask_indices.unsqueeze(-1).expand(-1, -1, x.size(-1))

    def toggle_embeddings(self):
        if not self.additional_patch_size:
            print('toggle_embeddings() called but no additional patch size provided! Ignoring call.')
            return
        self.to_patch_embedding, self.to_patch_embedding_b = self.to_patch_embedding_b, self.to_patch_embedding
        self.pos_embedding, self.pos_embedding_b = self.pos_embedding_b, self.pos_embedding

    def get_patch_coordinates(self, H, W, device=None):
        num_h = H // self.patch_height
        num_w = W // self.patch_width

        # make grid of indices
        ys, xs = torch.meshgrid(
            torch.arange(num_h, device=device),
            torch.arange(num_w, device=device),
            indexing='ij'
        )

        # flatten into [P, 2]
        coords = torch.stack([ys, xs], dim=-1).reshape(-1, 2)
        return coords  # shape [P, 2]

    def _make_embeddings(self, patch_height, patch_width, patch_dim, dim, image_height, image_width):
        to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        pos_embedding = posemb_sincos_2d(
            h=image_height // patch_height,
            w=image_width // patch_width,
            dim=dim,
        )

        return to_patch_embedding, pos_embedding