'''
Modified from the myna repository https://github.com/ghost-signal/myna
'''

from models.PositionalEmbeddings import *
from torch import nn
import matplotlib.pyplot as plt
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
        if not (rope_on_x or rope_on_y):
            raise ValueError("At least one axis must be enabled for RoPE.")

        if dim_head % 2 != 0:
            raise ValueError("dim_head must be even for rotary embeddings.")

        self.dim_head = dim_head
        self.rope_on_x = rope_on_x
        self.rope_on_y = rope_on_y

        self.axis_slices = []
        if rope_on_x and rope_on_y:
            if dim_head % 4 != 0:
                raise ValueError("dim_head must be divisible by 4 when both axes are enabled.")
            half = dim_head // 2
            self.axis_slices.append(("x", slice(0, half)))
            self.axis_slices.append(("y", slice(half, dim_head)))
        elif rope_on_x:
            self.axis_slices.append(("x", slice(0, dim_head)))
        elif rope_on_y:
            self.axis_slices.append(("y", slice(0, dim_head)))

        self.rope_base = rope_base

        for axis_name, axis_slice in self.axis_slices:
            axis_dim = axis_slice.stop - axis_slice.start
            half = axis_dim // 2

            exp = torch.arange(0, half).float() / half
            self.register_buffer(f"rope_exp_{axis_name}", exp)

    def _axis_cos_sin(self, coords_axis, axis_name, tempo_scale=None):
        coord = coords_axis.float()  # [b, n]
        exp = getattr(self, f"rope_exp_{axis_name}")  # [d/2]

        # base exponent
        log_base = math.log(self.rope_base)

        if axis_name == "x" and tempo_scale is not None:
            # tempo_scale: [b] or [b, n]
            if tempo_scale.dim() == 2:
                tempo_scale = tempo_scale.unsqueeze(-1)  # [b, n, 1]
            else:
                tempo_scale = tempo_scale.view(-1, 1, 1)

            inv_freq = torch.exp(
                -(exp.view(1, 1, -1) * log_base) * tempo_scale
            )
        else:
            inv_freq = torch.exp(
                -(exp.view(1, 1, -1) * log_base)
            )

        freqs = coord.unsqueeze(-1) * inv_freq
        cos = freqs.cos().unsqueeze(1)
        sin = freqs.sin().unsqueeze(1)
        return cos, sin

    def apply_rotary(t, cos, sin):
        t1, t2 = t[..., ::2], t[..., 1::2]
        return torch.cat(
            [t1 * cos - t2 * sin,
             t1 * sin + t2 * cos],
            dim=-1
        )

    def _apply_axis(self, tensor, axis_slice, cos, sin):
        left = tensor[..., :axis_slice.start]
        mid = tensor[..., axis_slice]
        right = tensor[..., axis_slice.stop:]

        rotated = self.apply_rotary(mid, cos, sin)
        return torch.cat((left, rotated, right), dim=-1)

    def forward(self, q, k, coords, tempo_scale):
        if coords is None:
            return q, k

        if coords.shape[1] + 1 == q.shape[2]:
            q_cls, q_tokens = q[:, :, :1], q[:, :, 1:]
            k_cls, k_tokens = k[:, :, :1], k[:, :, 1:]
        else:
            q_cls = k_cls = None
            q_tokens, k_tokens = q, k

        for axis_name, axis_slice in self.axis_slices:
            axis_index = 0 if axis_name == "x" else 1
            if coords.shape[-1] <= axis_index:
                continue

            axis_coords = coords[..., axis_index]
            cos, sin = self._axis_cos_sin(
                axis_coords,
                axis_name,
                tempo_scale if axis_name == "x" else None
            )
            q_tokens = self._apply_axis(q_tokens, axis_slice, cos, sin)
            k_tokens = self._apply_axis(k_tokens, axis_slice, cos, sin)

        if q_cls is not None:
            q = torch.cat([q_cls, q_tokens], dim=2)
            k = torch.cat([k_cls, k_tokens], dim=2)
        else:
            q, k = q_tokens, k_tokens

        return q, k
    

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64,
                 rope_base=-1, rope_on_x=False, rope_on_y=False,
                 predict_tempo=False):
        super().__init__()

        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)

        self.rotary = Rotary2D(
            dim_head=dim_head,
            rope_on_x=rope_on_x,
            rope_on_y=rope_on_y,
            rope_base=rope_base
        ) if rope_on_x or rope_on_y else None

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        # ===== Latent tempo / rhythm =====
        self.use_latent_tempo = False
        self.predict_tempo = False
        self.max_log_speed = 1.5

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

            elif predict_tempo == "latent":

                self.num_rhythm_heads = heads // 2  # e.g. half rhythmic
                self.rhythm_head_mask = torch.zeros(heads, dtype=torch.bool)
                self.rhythm_head_mask[:self.num_rhythm_heads] = True
                self.register_buffer("rhythm_head_mask", self.rhythm_head_mask)

                self.use_latent_tempo = True

                self.rhythm_token = nn.Parameter(torch.randn(1, 1, dim))

                # project token interaction → scalar latent
                self.rhythm_proj = nn.Linear(dim, dim, bias=False)
                # learnable phase offset (beat alignment)
                self.rope_offset = nn.Parameter(torch.zeros(1))


    def forward(self, x, alibi_bias=None, coords=None, mask=None):
        B, N, D = x.shape

        if self.predict_tempo == "latent":
            rhythm = self.rhythm_token.expand(B, -1, -1)
            x = torch.cat([rhythm, x], dim=1)

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        tempo_scale = None

        if self.predict_tempo:
            if self.predict_tempo == "latent":

                # latent tempo per token
                rhythm_state = F.normalize(x[:, 0], dim=-1)
                token_states = x[:, 1:]

                proj_tokens = self.rhythm_proj(token_states)
                z = torch.einsum('bnd,bd->bn', proj_tokens, rhythm_state)

                z = torch.tanh(z) * self.max_log_speed
                delta_t = torch.exp(z)
                tempo_coords = torch.cumsum(delta_t, dim=1)

                tempo_coords = torch.cat(
                    [torch.zeros(B, 1, device=x.device), tempo_coords],
                    dim=1
                )

                if coords is not None and coords.shape[1] + 1 == x.shape[1]:
                    tempo_scale = tempo_coords[:, 1:]  # remove CLS
                else:
                    tempo_scale = tempo_coords
            else:
                without_cls = x[:, 1:] if coords.shape[1] + 1 == x.shape[1] else x
                tempo_scale = self.tempo_head(without_cls)
                if self.predict_tempo == "MLP":
                    tempo_scale = tempo_scale.squeeze(2)

        if self.rotary is not None:
            rhythm_heads = self.rhythm_head_mask
            non_rhythm_heads = ~rhythm_heads

            # rhythm-aware heads
            if rhythm_heads.any():
                q_r, k_r = self.rotary(
                    q[:, rhythm_heads],
                    k[:, rhythm_heads],
                    coords,
                    tempo_scale
                )
                q[:, rhythm_heads] = q_r
                k[:, rhythm_heads] = k_r

            # non-rhythm heads
            if non_rhythm_heads.any():
                pass

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if alibi_bias is not None:
            dots = dots + alibi_bias

        if mask is not None:
            key_mask = mask[:, None, None, :]
            dots = dots.masked_fill(~key_mask, -1e8)

            query_mask = mask[:, None, :, None]
            dots = dots.masked_fill(~query_mask, -1e8)

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out[:, 1:] if self.use_latent_tempo else out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, rope_on_x=False, rope_on_y=False, alibi_on_x=False, alibi_on_y=False, clamping=None, rope_base=-1, predict_tempo=False):
        super().__init__()

        if alibi_on_x or alibi_on_y:
            self.alibi_2d = Alibi2DBias(heads, alibi_on_x=alibi_on_x, alibi_on_y=alibi_on_y, clamping=clamping)
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
            x = attn(x, alibi_bias, coords, mask) + x

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

        if mask is not None:
            mask = self.mask_to_patch_mask(mask).any(dim=-1)
            mask = mask.unsqueeze(1).expand(-1, 8, -1)
            mask = mask.reshape(B, -1)

        if self.use_sinusoidal:
            shape = x.shape
            x += self.pos_embedding[:shape[1],:].expand(B, -1, -1).to(device, dtype=x.dtype)

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
            unmasked = self.mask_inputs(x, mask, self.mask_ratio, device)
            x = x.gather(1, unmasked.unsqueeze(-1).expand(-1, -1, x.size(-1)))
            mask = mask.gather(1, unmasked)

            if self.needs_coordinates:
                coordinates = coordinates.gather(1, unmasked.unsqueeze(-1).expand(-1, -1, 2))

        B, P, F = x.shape

        if self.cls_token is not None:
            cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=B)
            x = torch.cat((cls_tokens, x), dim=1)

            # if self.needs_coordinates:
            #     cls_coord = torch.zeros(
            #         (B, 1, 2),
            #         device=device,
            #         dtype=coordinates.dtype
            #     )
            #     coordinates = torch.cat((cls_coord, coordinates), dim=1)

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

    def mask_inputs(self, x, mask, mask_ratio, device):
        B, N, _ = x.shape

        n_mask = int(mask_ratio * N)
        n_keep = max(1, N - n_mask)

        # Per-sample random permutations
        indices = torch.stack([
            torch.randperm(N, device=device)
            for _ in range(B)
        ])

        # Keep the last n_keep tokens for each sample
        unmasked_indices = indices[:, n_mask:n_mask + n_keep]

        return unmasked_indices

    def mask_to_patch_mask(self, mask: torch.Tensor) -> torch.Tensor:
        # (B, W)

        B, W = mask.shape
        ph, pw = self.patch_height, self.patch_width

        assert W % pw == 0, \
            "Mask spatial dimensions must be divisible by patch size"

        # Patchify exactly like the image
        patch_mask = rearrange(
            mask,
            "b (w pw) -> b (w) (pw)",
            pw=pw
        )

        return patch_mask


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