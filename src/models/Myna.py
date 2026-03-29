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
    def __init__(self, dim_head, rope_on_x=True, rope_on_y=True, rope_base=8192.0):
        super().__init__()
        if not (rope_on_x or rope_on_y):
            raise ValueError("At least one axis must be enabled for RoPE.")
        if dim_head % 2 != 0:
            raise ValueError("dim_head must be even for rotary embeddings.")

        self.dim_head = dim_head
        self.rope_on_x = rope_on_x
        self.rope_on_y = rope_on_y
        self.rope_base = rope_base

        # Determine slices for each axis
        self.axis_slices = []
        if rope_on_x and rope_on_y:
            if dim_head % 4 != 0:
                raise ValueError("dim_head must be divisible by 4 when both axes are enabled.")
            half = dim_head // 2
            self.axis_slices = [("x", slice(0, half)), ("y", slice(half, dim_head))]
        elif rope_on_x:
            self.axis_slices = [("x", slice(0, dim_head))]
        elif rope_on_y:
            self.axis_slices = [("y", slice(0, dim_head))]

        # Precompute exp for each axis
        for axis_name, axis_slice in self.axis_slices:
            axis_dim = axis_slice.stop - axis_slice.start
            half = axis_dim // 2
            exp = torch.arange(0, half).float() / half
            self.register_buffer(f"rope_exp_{axis_name}", exp)

    def _axis_cos_sin(self, coords_axis, axis_name):
        exp = getattr(self, f"rope_exp_{axis_name}")
        log_base = math.log(self.rope_base)
        inv_freq = torch.exp(-exp * log_base)  # [half_dim]

        # coords_axis: [..., axis_size, 1] * [half_dim] -> [..., axis_size, half_dim]
        freqs = coords_axis.unsqueeze(-1) * inv_freq
        cos = freqs.cos()
        sin = freqs.sin()
        return cos, sin

    def apply_rotary(self, t, cos, sin):
        # t: [..., dim], dim even
        t1, t2 = t[..., ::2], t[..., 1::2]
        return torch.cat([t1 * cos - t2 * sin, t1 * sin + t2 * cos], dim=-1)

    def _apply_axis(self, tensor, axis_slice, cos, sin, axis_dim):
        left = tensor[..., :axis_slice.start]
        mid = tensor[..., axis_slice]
        right = tensor[..., axis_slice.stop:]

        # Move the rotation axis to last dim for broadcasting
        mid_perm = mid.transpose(axis_dim, -2)  # [..., axis_size, F_slice]
        rotated = self.apply_rotary(mid_perm, cos, sin)
        rotated = rotated.transpose(axis_dim, -2)
        return torch.cat([left, rotated, right], dim=-1)

    def forward(self, x, coords):
        if coords is None:
            return x

        out = x
        for axis_name, axis_slice in self.axis_slices:
            axis_index = 0 if axis_name == "x" else 1
            axis_coords = coords[..., axis_index]  # [B, H, W]
            cos, sin = self._axis_cos_sin(axis_coords, axis_name)  # [B, H, W, half_dim]

            # Apply along the corresponding axis
            out = self._apply_axis(out, axis_slice, cos, sin, axis_index)

        return out
    

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
            # rhythm_heads = self.rhythm_head_mask
            # non_rhythm_heads = ~rhythm_heads
            #
            # # rhythm-aware heads
            # if rhythm_heads.any():
            #     q_r, k_r = self.rotary(
            #         q[:, rhythm_heads],
            #         k[:, rhythm_heads],
            #         coords,
            #         tempo_scale
            #     )
            #     q[:, rhythm_heads] = q_r
            #     k[:, rhythm_heads] = k_r
            #
            # # non-rhythm heads
            # if non_rhythm_heads.any():
            #     pass

            q, k = self.rotary(q, k, coords, tempo_scale)

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
    def __init__(self, dim, depth, heads, dim_head, mlp_dim,
                 use_rope_x=False, use_rope_y=False,
                 alibi_on_x=False, alibi_on_y=False,
                 clamping=None, rope_base=-1,
                 learned_alibi_slopes=False,
                 predict_tempo=False):

        super().__init__()

        if alibi_on_x or alibi_on_y:
            self.alibi_2d = Alibi2DBias(heads, alibi_on_x=alibi_on_x, alibi_on_y=alibi_on_y, clamping=clamping, learned_alibi_slopes=learned_alibi_slopes)
        else:
            self.alibi_2d = None

        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, rope_base=rope_base, rope_on_x=use_rope_x, rope_on_y=use_rope_y, predict_tempo=predict_tempo),
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

class StaticEmbeddings(nn.Module):
    def __init__(self, shape, image_height, image_width, patch_height, patch_width, d_model, use_sinusoidal_x=False,
                 use_sinusoidal_y=False, use_sinusoidal_raster=False, use_learned_encoding_y=False,
                 use_learned_encoding_x=False):

        super().__init__()

        self.height_in_patches = image_height // patch_height
        self.width_in_patches = image_width // patch_width

        self.use_learned_encoding_y = use_learned_encoding_y
        self.use_learned_encoding_x = use_learned_encoding_x
        self.use_sinusoidal_x = use_sinusoidal_x
        self.use_sinusoidal_y = use_sinusoidal_y
        self.use_sinusoidal_raster = use_sinusoidal_raster

        if use_learned_encoding_y:
            self.y_pos_embedding = nn.Parameter(torch.zeros(1, self.height_in_patches, 1, d_model))

        if use_learned_encoding_x:
            self.x_pos_embedding = nn.Parameter(torch.zeros(1, 1, self.width_in_patches, d_model))

        if use_sinusoidal_x and use_sinusoidal_y:
            self.sinusoidal_embeddings = self.generate_sinusoidal_2d(self.height_in_patches, self.width_in_patches, d_model).unsqueeze(0)
        elif use_sinusoidal_x:
            x_positions = torch.arange(self.width_in_patches, dtype=torch.float32).unsqueeze(1)
            x_emb = self.generate_sinusoidal_1d(x_positions, d_model)
            self.sinusoidal_embeddings = x_emb.unsqueeze(0).expand(self.height_in_patches, -1, -1).unsqueeze(0)
        elif use_sinusoidal_y:
            y_positions = torch.arange(self.height_in_patches, dtype=torch.float32).unsqueeze(1)
            y_emb = self.generate_sinusoidal_1d(y_positions, d_model)
            self.sinusoidal_embeddings = y_emb.unsqueeze(1).expand(-1, self.width_in_patches, -1).unsqueeze(0)
        elif use_sinusoidal_raster:
            positions = torch.arange(self.height_in_patches * self.width_in_patches, dtype=torch.float32).unsqueeze(1)
            raster_emb = self.generate_sinusoidal_1d(positions, d_model)
            self.sinusoidal_embeddings = raster_emb.view(self.height_in_patches, self.width_in_patches, -1).unsqueeze(0)

        self.shape = shape

    def forward(self, x):
        if self.use_learned_encoding_y:
            x += self.y_pos_embedding

        if self.use_learned_encoding_x:
            x += self.x_pos_embedding

        if self.use_sinusoidal_x or self.use_sinusoidal_y or self.use_sinusoidal_raster:
            x += self.sinusoidal_embeddings

        return x

    def generate_sinusoidal_2d(self, height, width, embed_dim):
        assert embed_dim % 2 == 0, "Embedding dimension must be divisible by 2"
        embed_dim = embed_dim // 2

        # Generate x positions
        x_positions = torch.arange(width, dtype=torch.float32).unsqueeze(1)
        x_embedding = self._generate_sinusoidal_1d(x_positions, embed_dim)

        # Generate y positions
        y_positions = torch.arange(height, dtype=torch.float32).unsqueeze(1)
        y_embedding = self._generate_sinusoidal_1d(y_positions, embed_dim)

        # Expand and combine to H x W x E
        x_emb_expanded = x_embedding.unsqueeze(0).expand(height, -1, -1)
        y_emb_expanded = y_embedding.unsqueeze(1).expand(-1, width, -1)

        pos_embedding = torch.cat([x_emb_expanded, y_emb_expanded], dim=-1)
        return pos_embedding

    def _generate_sinusoidal_1d(self, positions, dim):
        device = positions.device
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32, device=device) * -(torch.log(torch.tensor(10000.0)) / dim))
        pe = torch.zeros(positions.size(0), dim, device=device)
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        return pe  # N x E//2


class Myna(nn.Module):
    def __init__(self,
        image_size,
        channels = 1,
        patch_size = (16, 16),
        latent_space = 128,
        d_model = 384,
        depth = 12,
        heads = 6,
        mlp_dim = 1536,
        mask_ratio = 0.0,
        dim_head=64,
        use_rope_x = False,
        use_rope_y = False,
        use_alibi_x = False,
        use_alibi_y = False,
        use_learned_alibi_slopes = False,
        rope_base = 4096,
        latent_projection_method = "cls",
        use_sinusoidal_x = False,
        use_sinusoidal_y = False,
        use_sinusoidal_raster = True,
        use_learned_encoding_y = False,
        use_learned_encoding_x = False,
        use_rope_double_frequency = False,
        use_cls = False, clamping = None, predict_tempo = False):

        super().__init__()

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        self.patch_height = patch_height
        self.patch_width = patch_width

        self.num_patches_x = image_width // patch_width
        self.num_patches_y = image_height // patch_height
        self.num_patches = self.num_patches_x * self.num_patches_y

        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = self.make_embedding_projection(patch_height, patch_width, patch_dim, d_model)

        self.pos_embedding = StaticEmbeddings((1, self.num_patches, patch_dim), image_height, image_width, patch_height, patch_width, d_model,
                                                             use_sinusoidal_x=use_sinusoidal_x,
                                                             use_sinusoidal_y=use_sinusoidal_y,
                                                             use_sinusoidal_raster=use_sinusoidal_raster,
                                                             use_learned_encoding_y=use_learned_encoding_y,
                                                             use_learned_encoding_x=use_learned_encoding_x)

        self.use_rope_x = use_rope_x,
        self.use_rope_y = use_rope_y,
        self.use_alibi_x = use_alibi_x,
        self.use_alibi_y = use_alibi_y,
        self.use_learned_alibi_slopes = use_learned_alibi_slopes,
        self.rope_base = rope_base,
        self.latent_projection_method = latent_projection_method,
        self.use_sinusoidal_x = use_sinusoidal_x,
        self.use_sinusoidal_y = use_sinusoidal_y,
        self.use_sinusoidal_raster = use_sinusoidal_raster,
        self.use_learned_encoding_y = use_learned_encoding_y,
        self.use_learned_encoding_x = use_learned_encoding_x,
        self.use_rope_double_frequency = use_rope_double_frequency,

        self.needs_coordinates = use_rope_x or use_rope_y or use_alibi_x or use_alibi_y

        self.transformer = Transformer(d_model, depth, heads, dim_head, mlp_dim, clamping=clamping,
                                       rope_on_x=use_rope_x, rope_on_y=use_rope_y, alibi_on_x=use_alibi_x, alibi_on_y=use_alibi_y,
                                       rope_base=rope_base, predict_tempo=predict_tempo)

        self.transformer = Transformer(d_model, depth, heads, dim_head, mlp_dim, clamping=clamping,
                                       use_rope_x=self.use_rope_x,
                                       use_rope_y=self.use_rope_y,
                                       rope_base=self.rope_base,
                                       learned_alibi_slopes = self.learned_alibi_slopes,
                                       use_rope_double_frequency=self.use_rope_double_frequency,
                                       alibi_on_x=use_alibi_x,
                                       alibi_on_y=use_alibi_y)

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

        x += self.pos_embedding(x)

        if self.needs_coordinates:
            coordinates = self.get_patch_coordinates(H, W).expand(B, -1, -1).to(device)
        else:
            coordinates = None

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

    def make_embedding_projection(self, patch_height, patch_width, patch_dim, dim):
        to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        return to_patch_embedding

