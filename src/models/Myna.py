'''
Modified from the myna repository https://github.com/ghost-signal/myna
'''

from models.PositionalEmbeddings import *
from torch import nn
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
import torch
import torch.nn as nn

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class TempoEstimatorCNN(nn.Module):
    def __init__(self, d_model, hidden=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, hidden, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
        )

        self.head = nn.Sequential(
            nn.Conv1d(hidden, hidden // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden // 2, 1, kernel_size=1),
        )

        # small epsilon to keep tempo positive and non-zero
        self.eps = 1e-3

    def forward(self, x):
        B, N, D = x.shape
        x_t = x.transpose(1, 2)

        h = self.conv(x_t)
        h = self.head(h)

        # interpolate back to original N in case convs changed temporal resolution
        h = F.interpolate(h, size=N, mode='linear', align_corners=False)
        h = h.squeeze(1)

        # make positive and stable
        tempo = F.softplus(h) + self.eps

        return tempo

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


class Rotary2D(nn.Module):
    def __init__(self, dim_head, rope_on_x=False, rope_on_y=False, rope_base=8192.0):
        super().__init__()
        self.dim_head = dim_head
        self.rope_on_x = rope_on_x
        self.rope_on_y = rope_on_y
        
        assert dim_head % 2 == 0, "dim_head must be even for rotary"
        half = dim_head // 2
        inv_freq = 1.0 / (rope_base ** (torch.arange(0, half).float() / half))
        self.register_buffer("inv_freq", inv_freq)
        
    def get_cos_sin(self, coords, tempo_scale):
        B, N = coords.shape[:2]
        x = coords[..., 0].float()
        inv_freq = self.inv_freq.to(x.device)

        # Apply tempo scaling — allow broadcast
        #if tempo_scale.ndim == 1:
            #tempo_scale = tempo_scale.unsqueeze(0).expand(B, N)
                    
        adjusted = x
        if tempo_scale is not None:
             adjusted *= tempo_scale
        
        freqs_x = torch.einsum('bn,d->bnd', adjusted, inv_freq)

        if self.rope_on_y and coords.shape[-1] > 1:
            y = coords[..., 1].float()
            freqs_y = torch.einsum('bn,d->bnd', y, inv_freq)
            freqs = freqs_x + freqs_y
        else:
            freqs = freqs_x

        cos = freqs.cos().unsqueeze(1)
        sin = freqs.sin().unsqueeze(1)
        return cos, sin

    def apply_rotary(self, t, cos, sin):
        t1, t2 = t[..., ::2], t[..., 1::2]
        return torch.cat(
            [t1 * cos - t2 * sin,
             t1 * sin + t2 * cos],
            dim=-1
        )

    def forward(self, q, k, coords, tempo_scale):
        if coords is not None and coords.shape[1] + 1 == q.shape[2]:
            # Split CLS token
            q_cls, q_patches = q[:, :, :1], q[:, :, 1:]
            k_cls, k_patches = k[:, :, :1], k[:, :, 1:]

            cos, sin = self.get_cos_sin(coords, tempo_scale)
            q_patches = self.apply_rotary(q_patches, cos, sin)
            k_patches = self.apply_rotary(k_patches, cos, sin)

            q = torch.cat([q_cls, q_patches], dim=2)
            k = torch.cat([k_cls, k_patches], dim=2)
        else:
            cos, sin = self.get_cos_sin(coords, tempo_scale)
            q = self.apply_rotary(q, cos, sin)
            k = self.apply_rotary(k, cos, sin)

        return q, k
    

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, rope_base=-1, rope_on_x=False, rope_on_y=False, predict_tempo=False):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)

        self.rotary = Rotary2D(dim_head=dim_head, rope_on_x=rope_on_x, rope_on_y=rope_on_y, rope_base=rope_base) if rope_on_x or rope_on_y else None

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        
        self.predict_tempo = predict_tempo
        
        if predict_tempo:
            if predict_tempo == "CNN":
                self.tempo_head = TempoEstimatorCNN(dim)
                
                nn.init.constant_(self.tempo_head.head[-1].weight, 0.0)
                nn.init.constant_(self.tempo_head.head[-1].bias, 0.7)
                
            elif predict_tempo == "MLP":
                self.tempo_head = nn.Sequential(
                    nn.Linear(dim, dim // 2),
                    nn.GELU(),
                    nn.Linear(dim // 2, 1),
                    nn.Softplus()
                )
                
                nn.init.constant_(self.tempo_head[-2].weight, 0.0)
                nn.init.constant_(self.tempo_head[-2].bias, 0.7)


    def forward(self, x, alibi_bias=None, coords=None, mask=None):
        B, N, D = x.shape
        tempo_scale = None
        
        if self.predict_tempo:            
            without_cls = x[:, 1:] if coords.shape[1] + 1 == x.shape[1] else x
            tempo_scale = self.tempo_head(without_cls)
            if self.predict_tempo == "MLP":
                tempo_scale = tempo_scale.squeeze(2)

            #tempo_scale = self.tempo_head(x.mean(dim=1))
            #tempo_scale = tempo_scale + 1e-4

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        if hasattr(self, "rotary") and self.rotary:
            q, k = self.rotary(q, k, coords, tempo_scale)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if alibi_bias is not None:
            dots = dots + alibi_bias
            
        if mask is not None:
            attn_mask = mask.unsqueeze(1).unsqueeze(2)
            dots = dots.masked_fill(~attn_mask, float('-inf'))

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, rope_on_x=False, rope_on_y=False, alibi_on_x=False, alibi_on_y=False, clamping=None, rope_base=-1, predict_tempo=False):
        super().__init__()

        if alibi_on_x or alibi_on_y:
            self.alibi_2d = Alibi2DBias(heads, alibi_on_x=False, alibi_on_y=False, clamping=clamping)
        else:
            self.alibi_2d = None

        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, rope_base=rope_base, rope_on_x=rope_on_x, rope_on_y=rope_on_y, predict_tempo=predict_tempo),
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

    def forward(self, x, coords=None, cls=True, mask=None):
        if self.alibi_2d is not None:
            alibi_bias = self.alibi_2d(coords)
            alibi_bias = self.compute_alibi_with_cls(alibi_bias, has_cls=cls)
        else:
            alibi_bias = None

        for attn, ff in self.layers:
            x = attn(x, alibi_bias, coords=coords, mask=mask) + x
            x = ff(x) + x

        return self.norm(x)


class Myna(nn.Module):
    def __init__(self, *, image_size, patch_size, latent_space, d_model, depth, heads, mlp_dim, channels=3, dim_head=64,
                 mask_ratio: float = 0.0, use_cls=False, clamping=None, rope_base=512, predict_tempo=False,
                 use_sinusoidal=False, use_y_emb=False, use_rope_x=False, use_rope_y=False, use_alibi_x=False, use_alibi_y=False):
        super().__init__()

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        self.patch_height = patch_height
        self.patch_width = patch_width

        self.num_patches_y = image_height // patch_height

        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding, self.pos_embedding = self._make_embeddings(
            patch_height, patch_width, patch_dim, d_model, image_height, image_width
        )

        self.use_sinusoidal = use_sinusoidal
        self.use_y_emb = use_y_emb

        if not self.use_sinusoidal:
            self.pos_embedding = None

        if self.use_y_emb:
            self.y_pos_embedding = nn.Parameter(torch.zeros(image_height // patch_height, d_model))

        self.needs_coordinates = use_rope_x or use_rope_y or use_alibi_x or use_alibi_y

        self.transformer = Transformer(d_model, depth, heads, dim_head, mlp_dim, clamping=clamping,
                                       rope_on_x=use_rope_x, rope_on_y=use_rope_y, alibi_on_x=use_alibi_x, alibi_on_y=use_alibi_y,
                                       rope_base=rope_base, predict_tempo=predict_tempo)

        self.to_latent = nn.Identity()

        self.cls_token = torch.nn.Parameter(torch.randn(1, 1, d_model)) if use_cls else None

        self.mask_ratio = mask_ratio

        self.linear_head = nn.Linear(d_model, latent_space)

    def forward(self, img, mask=None):
        device = img.device

        B, _, H, W = img.shape

        x = self.to_patch_embedding(img)

        if self.use_sinusoidal:
            x += self.pos_embedding.to(device, dtype=x.dtype)

        if self.needs_coordinates:
            coordinates = self.get_patch_coordinates(H, W).expand(B, -1, -1).to(device)
        else:
            coordinates = None

        if self.use_y_emb:
            B, N, D = x.shape
            H = self.num_patches_y
            W = N // H

            y_emb = self.y_pos_embedding[:H].to(device, dtype=x.dtype)
            y_emb = y_emb.unsqueeze(1).repeat(1, W, 1)
            y_emb = y_emb.reshape(1, H * W, D).repeat(B, 1, 1)

            x = x + y_emb

        if self.mask_ratio > 0.0:
            unmasked = self.mask_inputs(x, self.mask_ratio, device)
            x = x.gather(1, unmasked.unsqueeze(-1).expand(-1, -1, x.size(-1)))

            if self.needs_coordinates:
                coordinates = coordinates.gather(1, unmasked.unsqueeze(-1).expand(-1, -1, 2))
                
            if mask is not None:
                mask = mask.squeeze(1).gather(1, unmasked).to(device, dtype=torch.bool)

        B, P, F = x.shape

        if self.cls_token is not None:
            cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=B)
            x = torch.cat((cls_tokens, x), dim=1)
            
            if mask is not None:
                cls_mask = torch.ones((B, 1), dtype=torch.bool, device=device)
                mask = torch.cat((cls_mask, mask), dim=1)

        x = self.transformer(x, coords=coordinates, cls=self.cls_token is not None, mask=mask)

        if self.cls_token is not None:
            x = x[:, 0]
        else:
            if mask is not None:
                mask = mask.to(x.dtype)
                x = (x * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1e-6)
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
            indexing='ij'
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