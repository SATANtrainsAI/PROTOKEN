import torch
import torch.nn as nn
import torch.nn.functional as F



from rotary_embedding_torch import RotaryEmbedding
from einops import rearrange


from dataclasses import dataclass
from typing import Optional, Tuple, List
import os
import inspect


def exists(val):
    return val is not None

def print0(*args, **kwargs):
    # modified print that only prints from the master process
    # if this is not a distributed run, it's just a print
    if int(os.environ.get("RANK", 0)) == 0:
        print(*args, **kwargs)


# =============================================================================
# 1. LayerNorm
# =============================================================================
class LayerNorm(nn.Module):
    """
    A simple LayerNorm without bias by default, as used in many transformer blocks.
    """
    def __init__(self, ndim, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, eps=1e-5)

# =============================================================================
# 2. Multi-Head Self-Attention with Rotary Embedding (MLA)
# =============================================================================
class MLA(nn.Module):
    """
    Multi-head attention block for 2D patches flattened into sequences.
    Uses scaled_dot_product_attention (PyTorch 2.0).
    """
    def __init__(self, config):
        super().__init__()
        # Ensure latent_dim is divisible by n_head
        assert config.latent_dim % config.n_head == 0, "latent_dim must be divisible by n_head"

        # Project from (n_embd) -> (latent_dim * 3) to get Q, K, V
        self.c_attn = nn.Linear(config.n_embd, config.latent_dim * 3, bias=False)
        # Final projection back from latent_dim -> n_embd
        self.c_proj = nn.Linear(config.latent_dim, config.n_embd, bias=False)

        # Rotary embedding
        self.rotary_emb = RotaryEmbedding(dim=config.rope_dim)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.latent_dim = config.latent_dim

        # For VQ‐VAE, we typically do not want causal masking
        self.is_causal = config.is_causal  

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: shape (B, T, n_embd), T is the flattened 2D spatial dimension
        """
        B, T, _ = x.size()
        qkv = self.c_attn(x)  # (B, T, 3*latent_dim)
        q, k, v = qkv.split(self.latent_dim, dim=2)

        # Reshape into (B, heads, T, dim_per_head)
        q = q.view(B, T, self.n_head, self.latent_dim // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.latent_dim // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.latent_dim // self.n_head).transpose(1, 2)

        # Apply rotary embeddings to q and k
        q = self.rotary_emb.rotate_queries_or_keys(q)
        k = self.rotary_emb.rotate_queries_or_keys(k)

        # Scaled dot product attention (PyTorch 2.0+)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=self.is_causal
        )
        # Reshape back
        y = y.transpose(1, 2).contiguous().view(B, T, self.latent_dim)
        # Final linear back to n_embd
        y = self.c_proj(y)
        return y

# =============================================================================
# 3. MLP Block
# =============================================================================
class MLP(nn.Module):
    """
    Simple 2-layer MLP for feedforward portion of a transformer block.
    """
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.mlp_hidden_dim, bias=False)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(config.mlp_hidden_dim, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

# =============================================================================
# 4. Transformer Block
# =============================================================================
class Block(nn.Module):
    """
    A single Transformer block, consisting of:
      - LayerNorm
      - Multi-head Self-Attention
      - LayerNorm
      - MLP
    Each is a residual connection around LN+Attention or LN+MLP.
    """
    def __init__(self, config):
        super().__init__()
        self.ln1 = LayerNorm(config.n_embd, bias=False)
        self.mla = MLA(config)
        self.ln2 = LayerNorm(config.n_embd, bias=False)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mla(self.ln1(x))   # Self-attention
        x = x + self.mlp(self.ln2(x))  # Feedforward
        return x

# =============================================================================
# 5. Vision Transformer Encoder
# =============================================================================

class DownsampleStack(nn.Module):
    """
    Replaces the single patch_emb Conv2d with multiple smaller-stride conv layers.
    Overlaps by using kernel_size=3, stride=2, repeated log2(downscale_factor) times.
    """
    def __init__(self, in_channels: int, out_channels: int, factor: int):
        super().__init__()
        layers = []
        current_ch = in_channels
        f = factor
        while f > 1:
            layers.append(nn.Conv2d(current_ch, out_channels, kernel_size=2, stride=2))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.GELU())
            current_ch = out_channels
            f //= 2
        self.down = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)


class UpsampleStack(nn.Module):
    """
    Replaces the single patch_unemb ConvTranspose2d with multiple smaller-stride transposed conv layers.
    Each step doubles the spatial resolution (stride=2).
    """
    def __init__(self, in_channels: int, out_channels: int, factor: int):
        super().__init__()
        layers = []
        current_ch = in_channels
        f = factor
        while f > 1:
            layers.append(nn.ConvTranspose2d(current_ch, out_channels, kernel_size=2, stride=2))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.GELU())
            current_ch = out_channels
            f //= 2
        self.up = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


class VitEncoder(nn.Module):
    """
    Takes an image, downsamples it by 'downscale_factor' via a Conv2d,
    flattens the resulting feature map, then applies several Transformer blocks.
    Finally, reshapes back to (B, C, H, W).
    """
    def __init__(self, config, downscale_factor: int, in_channels: Optional[int] = None):
        super().__init__()
        if in_channels is None:
            in_channels = config.n_embd  # default

        # Patch embedding by conv with kernel_size = stride = downscale_factor
        self.patch_emb = DownsampleStack(
            in_channels=in_channels,
            out_channels=config.n_embd,
            factor=downscale_factor
        )
        # Stacked transformer blocks
        self.encoder_blocks = nn.Sequential(*[
            Block(config) for _ in range(config.n_layer_encoder)
        ])
    
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        img: (B, in_channels, H, W)
        returns: (B, n_embd, H//downscale, W//downscale)
        """
        x = self.patch_emb(img)  # shape: (B, n_embd, H/down, W/down)
        B, C, H, W = x.shape
        # Flatten (spatial) => (B, H*W, C)
        x = rearrange(x, 'b c h w -> b (h w) c')
        # Apply the Transformer
        x = self.encoder_blocks(x)
        # Reshape back
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        return x

# =============================================================================
# 6. Vision Transformer Decoder
# =============================================================================
class VitDecoder(nn.Module):
    """
    Takes a multi-code concatenation as input: shape (B, n_embd*(level+1), H, W).
    1) Projects down to n_embd
    2) Applies stacked transformer blocks
    3) ConvTranspose back up by 'upscale_factor'.
    """
    def __init__(self, config, upscale_factor: int, level: int, out_channels: Optional[int] = None):
        """
        :param level: the number of codes being concatenated minus 1.
                      For example, if we're concatenating 2 codes, level=1 => input channels = n_embd * 2.
        """
        super().__init__()
        if out_channels is None:
            out_channels = config.n_embd  # default to embedding dimension

        self.config = config
        self.level = level  # (level+1) codes are concatenated
        in_ch = config.n_embd * (level + 1)

        # Project from in_ch -> n_embd via 1x1 conv
        self.in_proj = nn.Conv2d(in_ch, config.n_embd, kernel_size=1)

        # Stacked transformer blocks
        self.decoder_blocks = nn.Sequential(*[
            Block(config) for _ in range(config.n_layer_decoder)
        ])

        # Transpose conv to "unpatch" with kernel_size = stride = upscale_factor
        self.patch_unemb = UpsampleStack(
            in_channels=config.n_embd,
            out_channels=out_channels,
            factor=upscale_factor
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, n_embd*(level+1), H, W)
        returns: (B, out_channels, H*upscale, W*upscale)
        """
        # 1) Project input to n_embd
        x = self.in_proj(x)  # shape: (B, n_embd, H, W)
        B, C, H, W = x.shape

        # 2) Flatten and run Transformer
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.decoder_blocks(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)

        # 3) Transpose conv up to original resolution
        x = self.patch_unemb(x)
        return x

# =============================================================================
# 7. Quantize (VQ-VAE Codebook with EMA) in NCHW
# =============================================================================
class Quantize(nn.Module):
    """
    Learns a discrete codebook with an exponential moving average update,
    as in the original VQ-VAE2 by Oord et al. or Rosinality's code.

    NOTE: We keep everything in NCHW for the commitment loss to avoid shape mismatches.
    """
    def __init__(self, config, in_channels: Optional[int] = None):
        super().__init__()
        self.codebook_dim = config.codebook_dim  # embedding dimension
        if in_channels is None:
            self.in_channels = config.q_channels
        else:
            self.in_channels = in_channels

        self.codebook_size = config.codebook_size  # number of entries
        self.decay = 0.99
        self.eps = 1e-5

        # 1x1 conv to go from in_channels -> codebook_dim
        self.conv_in = nn.Conv2d(self.in_channels, self.codebook_dim, 1)

        # Initialize the codebook: shape (codebook_dim, codebook_size)
        codebook = torch.randn(self.codebook_dim, self.codebook_size, dtype=torch.float32)
        self.register_buffer("codebook", codebook)
        self.register_buffer("cluster_size", torch.zeros(self.codebook_size, dtype=torch.float32))
        self.register_buffer("codebook_avg", codebook.clone())

    @torch.autocast("cuda", enabled=False)
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param x: (B, in_channels, H, W)
        :return: quantized, diff, codebook_indices
           quantized: (B, codebook_dim, H, W)
           diff: scalar codebook commitment loss
           codebook_indices: (B, H, W), the argmax index per spatial location
        """
        # 1) Project to codebook_dim, staying in NCHW
        x_projected = self.conv_in(x.float())  # => (B, codebook_dim, H, W)

        B, D, H, W = x_projected.shape

        # 2) Flatten spatial dims => (B*H*W, D)
        x_reshaped = rearrange(x_projected, 'b d h w -> (b h w) d')

        # 3) Compute distances to each codebook entry, shape => (B*H*W, codebook_size)
        #    dist(i, j) = || x_i - e_j ||^2
        codebook_t = self.codebook  # shape (codebook_dim, codebook_size)
        dist = (
            x_reshaped.pow(2).sum(dim=1, keepdim=True)
            - 2 * (x_reshaped @ codebook_t)
            + codebook_t.pow(2).sum(dim=0, keepdim=True)
        )  # (B*H*W, codebook_size)

        # 4) Find nearest codebook entry
        _, indices = (-dist).max(dim=1)          # (B*H*W,)
        indices = indices.view(B, H, W)          # (B, H, W)
        codebook_onehot = F.one_hot(indices, self.codebook_size).type(x_reshaped.dtype)
        codebook_onehot = rearrange(codebook_onehot, 'b h w c -> (b h w) c')  # (B*H*W, codebook_size)

        # 5) Lookup embedding from the codebook => shape (B, H, W, codebook_dim)
        quantized_hw_d = self.embed_code(indices)

        # 6) If training, do EMA updates
        if self.training:
            # sum of one-hot per codebook entry => cluster size
            codebook_onehot_sum = codebook_onehot.sum(dim=0)  # (codebook_size,)

            # sum of x_reshaped for each code => (codebook_dim, codebook_size)
            codebook_sum = x_reshaped.transpose(0, 1) @ codebook_onehot  # (D, codebook_size)

            # Update cluster size
            self.cluster_size.data.mul_(self.decay).add_(codebook_onehot_sum, alpha=1 - self.decay)
            # Update codebook averages
            self.codebook_avg.data.mul_(self.decay).add_(codebook_sum, alpha=1 - self.decay)

            # Normalize so each codebook vector is the average of all assigned vectors
            n = self.cluster_size.sum()
            cluster_size = (self.cluster_size + self.eps) / (n + self.codebook_size * self.eps) * n
            codebook_normalized = self.codebook_avg / cluster_size.unsqueeze(0)
            self.codebook.data.copy_(codebook_normalized)

        # 7) Compute commitment loss in NCHW
        #    quantized_hw_d is (B, H, W, D)
        #    let's transpose that to (B, D, H, W)
        quantized_nchw = rearrange(quantized_hw_d, 'b h w d -> b d h w')
        diff = (quantized_nchw.detach() - x_projected).pow(2).mean()

        # 8) Straight-through estimator
        #    final quantized = x_projected + (quantized - x_projected).detach()
        quantized_nchw = x_projected + (quantized_nchw - x_projected).detach()

        # 9) Return NCHW quantized
        return quantized_nchw, diff, indices

    def embed_code(self, embed_id: torch.Tensor) -> torch.Tensor:
        """
        embed_id: (B, H, W) of codebook indices in [0, codebook_size)
        returns: (B, H, W, codebook_dim)
        """
        # codebook: (codebook_dim, codebook_size)
        # F.embedding => shape (B*H*W, codebook_dim), we reshape to (B,H,W, D)
        out_flat = F.embedding(embed_id.view(-1), self.codebook.transpose(0, 1))
        out = rearrange(out_flat, '(b h w) d -> b h w d', b=embed_id.shape[0], h=embed_id.shape[1], w=embed_id.shape[2])
        return out



class RefinementHead(nn.Module):
    """
    A small transformer-based refinement step for the final (B,3,H,W) output,
    to help sharpen or denoise.
    """
    def __init__(self, config):
        super().__init__()
        # Project 3 -> n_embd
        self.in_proj = nn.Conv2d(3, config.n_embd, kernel_size=1)
        # A small stack of Transformer blocks (e.g. 2)
        self.refine_blocks = nn.Sequential(*[
            Block(config) for _ in range(2)
        ])
        # Project back n_embd -> 3
        self.out_proj = nn.Conv2d(config.n_embd, 3, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, H, W)
        returns: (B, 3, H, W) refined
        """
        B, C, H, W = x.shape
        r = self.in_proj(x)  # (B, n_embd, H, W)
        r = rearrange(r, 'b c h w -> b (h w) c')
        r = self.refine_blocks(r)  # Transform
        r = rearrange(r, 'b (h w) c -> b c h w', h=H, w=W)
        r = self.out_proj(r)  # back to 3 channels
        return r

# =============================================================================
# 8. Model Configuration
# =============================================================================
@dataclass
class ModelArgs:
    """
    Hyperparameters for the model.
    Adjust to your preference. Typically n_embd ~ 256..768,
    n_head ~ 4..16, and so on, depending on memory and resolution.
    """
    # Basic
    n_embd: int = 1024
    mlp_hidden_dim: int = int(1024 * 2)
    n_head: int = 16
    latent_dim: int = 256   # dimension for Q/K/V
    rope_dim: int = 8
    is_causal: bool = False  # Usually no causal mask for VQ-VAE

    # Codebook
    q_channels: int = 1024
    codebook_dim: int = 1024
    codebook_size: int = 2048

    # Depth
    n_layer_encoder: int = 4
    n_layer_decoder: int = 4

    beta: float = 0.25

# =============================================================================
# 9. Hierarchical VQ-VAE (Transformer-based)
# =============================================================================
class VQVAE(nn.Module):
    """
    A multi-level VQ-VAE-2 style model with Transformers at each stage.
    - Each stage: 
      1) Encode with VitEncoder
      2) Next stage uses the output of the previous stage
    - Decoding top-down in reverse order, each code conditioned on
      the corresponding encoder output + upsampled next-lower-level decoder output
    - Then quantize each stage's conditioning, feed into a VitDecoder
    """
    def __init__(self, config: ModelArgs, scaling_rates=[2, 2, 2, 2, 2, 2]):
        super().__init__()
        self.config = config
        self.scaling_rates = scaling_rates
        self.num_levels = len(scaling_rates)

        # ---------------------
        # Build encoders
        # ---------------------
        self.encoders = nn.ModuleList()
        for level in range(self.num_levels):
            # The first encoder sees 3-channel images
            in_ch = 3 if level == 0 else None
            self.encoders.append(
                VitEncoder(config, downscale_factor=scaling_rates[level], in_channels=in_ch)
            )

        # ---------------------
        # Build codebooks
        # ---------------------
        self.codebooks = nn.ModuleList()
        for level in range(self.num_levels):
            # The topmost level (level == num_levels - 1) has in_ch = n_embd
            # otherwise in_ch = 2 * n_embd (concatenate encoder_out + lower-level dec_out)
            if level == self.num_levels - 1:
                in_ch = config.n_embd
            else:
                in_ch = config.n_embd * 2
            self.codebooks.append(Quantize(config, in_channels=in_ch))

        # ---------------------
        # Build decoders
        # Decoding in top-down order: level=2 => coarsest => outputs n_embd
        #                             level=1 => outputs n_embd
        #                             level=0 => outputs 3
        # but we store them in normal order, so index=0 => factor=8 => final upsample
        # We'll just carefully pick out_channels = 3 if level==0, else n_embd
        # We'll feed (num_levels - level) codes into the decoder.
        # ---------------------
        self.decoders = nn.ModuleList()
        for level in range(self.num_levels):
            out_ch = 3 if level == 0 else config.n_embd
            codes_to_cat = (self.num_levels - level)
            self.decoders.append(
                VitDecoder(
                    config,
                    upscale_factor=scaling_rates[level],
                    level=(codes_to_cat - 1),
                    out_channels=out_ch
                )
            )

        self.refiner = RefinementHead(config)
        
        self.apply(self._init_weights)

        print0("number of parameters: %.2fM" % (self.get_num_params()/1e6,))
    
    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def forward(self, x: torch.Tensor):
        """
        :param x: (B, 3, H, W) input image
        :return: (recon, code_diffs, encoder_outputs, decoder_outputs, code_indices)
        """
        # ----------------------------------
        # 1) Bottom-up encoding
        # ----------------------------------
        encoder_outputs = []
        for level, encoder in enumerate(self.encoders):
            if level == 0:
                out = encoder(x)
            else:
                out = encoder(encoder_outputs[-1])
            encoder_outputs.append(out)  # shape: (B, n_embd, H/scale, W/scale)

        # ----------------------------------
        # 2) Top-down decoding
        # ----------------------------------
        code_diffs = []
        code_indices = []
        decoder_outputs = []  # store from coarse to fine
        code_outputs = []     # store the quantized codes from coarse to fine

        # We go in reversed order: self.num_levels-1 (coarsest) down to 0 (finest)
        for l in reversed(range(self.num_levels)):
            # Combine the encoder output with upsampled next-lower-level decoder output (if any)
            if len(decoder_outputs) == 0:
                # no finer-level decoder yet
                cond = encoder_outputs[l]
            else:
                # upsample the last decoder output to match this scale
                prev_dec = decoder_outputs[-1]
                target_size = encoder_outputs[l].shape[2:]  # (H, W) at current scale
                prev_dec_upsampled = F.interpolate(prev_dec, size=target_size, mode='bilinear', align_corners=False)
                # Cat => (B, 2*n_embd, H, W)
                cond = torch.cat([encoder_outputs[l], prev_dec_upsampled], dim=1)

            # Quantize the conditioning
            q, diff, inds = self.codebooks[l](cond)
            code_diffs.append(diff)
            code_indices.append(inds)

            # Also upsample all previously computed codes so they match this resolution
            upsampled_lower_codes = []
            for code_map in code_outputs:
                c_up = F.interpolate(code_map, size=cond.shape[2:], mode='bilinear', align_corners=False)
                upsampled_lower_codes.append(c_up)

            # Concatenate the new code q with all upsampled codes from even finer levels
            if len(upsampled_lower_codes) > 0:
                dec_input = torch.cat([q] + upsampled_lower_codes, dim=1)
            else:
                dec_input = q

            # Decode
            dec_out = self.decoders[l](dec_input)
            decoder_outputs.append(dec_out)
            code_outputs.append(q)

        # The final reconstruction is decoder_outputs[-1] (the last appended)
        reconstruction = decoder_outputs[-1]

        # Reverse lists if you prefer them from level=0..N-1
        # but it's optional. Right now they are in [coarsest -> finest] order
        reconstruction_refined = self.refiner(reconstruction)

        return reconstruction_refined, code_diffs, encoder_outputs, decoder_outputs, code_indices
    
    def loss(self, X, reconstruction, code_diffs):
        r_loss, l_loss = reconstruction.sub(X).pow(2).mean(), sum(code_diffs)
        Loss =  r_loss + self.config.beta * l_loss
        return Loss, r_loss, l_loss
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        
        param_dict = {pn: p for pn, p in self.named_parameters()}
        
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print0(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print0(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print0(f"using fused AdamW: {use_fused}")

        return optimizer
    
