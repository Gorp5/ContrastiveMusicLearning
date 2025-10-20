'''
Modified from the myna repository https://github.com/ghost-signal/myna
'''

from models.PositionalEmbeddings import *
from torch import nn
from einops.layers.torch import Rearrange
from einops import rearrange, repeat


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

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

    def forward(self, x, alibi_bias=None):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if alibi_bias is not None:
            dots = dots + alibi_bias

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.alibi_2d = Alibi2DBias(heads)
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head),
                FeedForward(dim, mlp_dim)
            ]))

    def compute_alibi_with_cls(self, alibi_bias_2d, has_cls: bool):
        if not has_cls:
            return alibi_bias_2d

        B, H, N, _ = alibi_bias_2d.shape
        device = alibi_bias_2d.device
        dtype = alibi_bias_2d.dtype

        # Create full zero tensor and append biases into lower right slot
        full = torch.zeros((B, H, N + 1, N + 1), device=device, dtype=dtype)
        full[:, :, 1:, 1:] = alibi_bias_2d
        return full

    def forward(self, x, coords=None, cls=True):
        if coords is not None:
            alibi_bias = self.alibi_2d(coords)
            alibi_bias = self.compute_alibi_with_cls(alibi_bias, has_cls=cls)
        else:
            alibi_bias = None

        for attn, ff in self.layers:
            x = attn(x, alibi_bias) + x
            x = ff(x) + x
        return self.norm(x)


class Myna(nn.Module):
    def __init__(self, *, image_size, patch_size, latent_space, d_model, depth, heads, mlp_dim, channels=3, dim_head=64,
                 additional_patch_size=None, mask_ratio: float = 0.0, use_cls=False, alibi=False):
        super().__init__()

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        self.patch_height = patch_height
        self.patch_width = patch_width

        self.alibi = alibi

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

        self.transformer = Transformer(d_model, depth, heads, dim_head, mlp_dim)

        self.to_latent = nn.Identity()

        self.cls_token = torch.nn.Parameter(torch.randn(1, 1, d_model)) if use_cls else None

        self.mask_ratio = mask_ratio

        self.linear_head = nn.Linear(d_model, latent_space)

    def forward(self, img):
        device = img.device

        B, _, H, W = img.shape

        x = self.to_patch_embedding(img)

        if self.alibi:
            coordinates = self.get_patch_coordinates(H, W).expand(B, -1, -1).to(device)
        else:
            x += self.pos_embedding.to(device, dtype=x.dtype)

        if self.mask_ratio > 0.0:
            unmasked = self.mask_inputs(x, self.mask_ratio, device)
            x = x.gather(1, unmasked.unsqueeze(-1).expand(-1, -1, x.size(-1)))

            if self.alibi:
                coordinates = coordinates.gather(1, unmasked.unsqueeze(-1).expand(-1, -1, 2))

        B, P, F = x.shape

        if self.cls_token is not None:
            cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=B)
            x = torch.cat((cls_tokens, x), dim=1)

        if self.alibi:
            x = self.transformer(x, coords=coordinates, cls=self.cls_token is not None)
        else:
            x = self.transformer(x)

        if self.cls_token is not None:
            x = x[:, 0]
        else:
            x = x.mean(dim=1)

        x = self.to_latent(x)
        return self.linear_head(x)


    def mask_inputs(self, x: torch.Tensor, mask_ratio: float, device: str):
        B, N, _ = x.shape
        n_masked = int(mask_ratio * N)

        indices = torch.stack([torch.randperm(N, device=device) for _ in range(B)])
        unmask_indices = indices[:, n_masked:]
        return unmask_indices


    def get_patch_coordinates(self, H, W, device=None):
        num_h = H // self.patch_height
        num_w = W // self.patch_width

        ys, xs = torch.meshgrid(
            torch.arange(num_h, device=device),
            torch.arange(num_w, device=device),
            indexing='i j'
        )

        coords = torch.stack([ys, xs], dim=-1).reshape(-1, 2)
        return coords

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