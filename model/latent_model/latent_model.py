import torch, math
from torch.utils.checkpoint import checkpoint
import numpy as np
import torch.nn as nn

# Assuming these are available in your local directories
from .standalone_hyena import HyenaOperator
from .mha import MultiheadAttention
from .layers import GaussianFourierProjection, TimestepEmbedder, FinalLayer
from .layers import gelu, modulate
from .ipa import InvariantPointAttention

def grad_checkpoint(func, args, checkpointing=False): 
    if checkpointing:
        def wrapper(*inputs):
            # Explicitly force the recomputation to use bfloat16
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                return func(*inputs)
        return checkpoint(wrapper, *args, use_reentrant=False)
    else:
        return func(*args)

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos): 
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  

    pos = pos.reshape(-1)  
    out = np.einsum('m,d->md', pos, omega)  

    emb_sin = np.sin(out)  
    emb_cos = np.cos(out)  

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  
    return emb 


class LatentMDGenModel(nn.Module):
    def __init__(self, args, latent_dim=7): # Defaulting latent_dim to 7 (3 translation, 4 quaternion)
        super().__init__()
        self.args = args

        # Main input embedding (N particles * T frames * 7 dims -> Embed Dim)
        self.latent_to_emb = nn.Linear(latent_dim, args.embed_dim) 
        
        # Conditioning on the initial frame (start_frames)
        self.cond_to_emb = nn.Linear(latent_dim, args.embed_dim)
        self.mask_to_emb = nn.Embedding(2, args.embed_dim)
        
        if args.box is not None:
            ipa_args = {
                'c_s': args.embed_dim,
                'c_z': 0,
                'c_hidden': args.ipa_head_dim,
                'no_heads': args.ipa_heads,
                'no_qk_points': args.ipa_qk,
                'no_v_points': args.ipa_v,
                'dropout': args.dropout,
                'box': args.box,
            }

        else:
            ipa_args = {
                'c_s': args.embed_dim,
                'c_z': 0,
                'c_hidden': args.ipa_head_dim,
                'no_heads': args.ipa_heads,
                'no_qk_points': args.ipa_qk,
                'no_v_points': args.ipa_v,
                'dropout': args.dropout,
            }

        if args.prepend_ipa: 
            self.aatype_to_emb = nn.Embedding(1, args.embed_dim) # all particles are the same
            self.ipa_layers = nn.ModuleList( 
                [
                    IPALayer( 
                        embed_dim=args.embed_dim,
                        ffn_embed_dim=4 * args.embed_dim,
                        mha_heads=args.mha_heads,
                        dropout=args.dropout,
                        use_rotary_embeddings=args.use_rope_l,
                        ipa_args=ipa_args
                    )
                    for _ in range(args.num_layers) 
                ]
            ) 

        self.layers = nn.ModuleList( 
            [
                LatentMDGenLayer( 
                    embed_dim=args.embed_dim,
                    ffn_embed_dim=4 * args.embed_dim,
                    mha_heads=args.mha_heads,
                    dropout=args.dropout,
                    hyena=args.hyena,
                    num_frames=args.num_frames,
                    use_rotary_embeddings_t=args.use_rope_t, # Pass T-specific flag
                    use_rotary_embeddings_l=args.use_rope_l, # Pass L-specific flag
                    use_time_attention=True,
                    ipa_args=ipa_args if args.interleave_ipa else None,
                )
                for _ in range(args.num_layers)
            ]
        )

        self.emb_to_latent = FinalLayer(args.embed_dim, latent_dim) 
        self.t_embedder = TimestepEmbedder(args.embed_dim) 

        if args.abs_pos_emb: 
            self.register_buffer('pos_embed', 
                                 nn.Parameter(torch.zeros(1, args.crop, args.embed_dim), requires_grad=False))
        if args.abs_time_emb:
            self.register_buffer('time_embed',
                                 nn.Parameter(torch.zeros(1, args.num_frames, args.embed_dim), requires_grad=False))

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init) 

        if self.args.prepend_ipa:
            for block in self.ipa_layers:
                nn.init.constant_(block.ipa.linear_out.weight, 0)
                nn.init.constant_(block.ipa.linear_out.bias, 0)

        if self.args.interleave_ipa:
            for block in self.layers:
                nn.init.constant_(block.ipa.linear_out.weight, 0)
                nn.init.constant_(block.ipa.linear_out.bias, 0)

        if self.args.abs_pos_emb:
            pos_embed = get_1d_sincos_pos_embed_from_grid(self.pos_embed.shape[-1], np.arange(self.args.crop)) 
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0)) 
        if self.args.abs_time_emb:
            time_embed = get_1d_sincos_pos_embed_from_grid(self.time_embed.shape[-1], np.arange(self.args.num_frames))
            self.time_embed.data.copy_(torch.from_numpy(time_embed).float().unsqueeze(0))

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02) 
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.layers:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.emb_to_latent.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.emb_to_latent.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.emb_to_latent.linear.weight, 0)
        nn.init.constant_(self.emb_to_latent.linear.bias, 0)

    def run_ipa(self, t, mask, start_frames):
        B, L = mask.shape
        x = torch.zeros(B, L, self.args.embed_dim, device=mask.device)
        # Assuming B and L are already defined from mask.shape
        aatype = torch.zeros(B, L, dtype=torch.long, device=mask.device)
        x = x + self.aatype_to_emb(aatype)
        for layer in self.ipa_layers:
            x = layer(x, t, mask, frames=start_frames)
        return x

    def forward(self, x, t, mask,
                 start_frames=None, 
                 x_cond=None,
                 x_cond_mask=None):
        # x is the noisy trajectory of shape (B, T, N, 7)
        x = self.latent_to_emb(x)  

        if self.args.abs_pos_emb:
            x = x + self.pos_embed 
        if self.args.abs_time_emb:
            x = x + self.time_embed[:, :, None] 

        if x_cond is not None:
            # x_cond is the clean first frame of shape (B, N, 7), broadcasted to T
            x = x + self.cond_to_emb(x_cond) + self.mask_to_emb(x_cond_mask)

        t = self.t_embedder(t * self.args.time_multiplier)[:, None] 

        if self.args.prepend_ipa: 
            x = x + self.run_ipa(t[:, 0], mask[:, 0], start_frames)[:, None] 

        for layer_idx, layer in enumerate(self.layers):
            x = grad_checkpoint(layer, (x, t, mask, start_frames), self.args.grad_checkpointing) 

        latent = self.emb_to_latent(x, t) 
        return latent

class AttentionWithRoPE(nn.Module):
    def __init__(self, *args, **kwargs): 
        super().__init__()
        self.attn = MultiheadAttention(*args, **kwargs)

    def forward(self, x, mask):
        x = x.transpose(0, 1) 
        x, _ = self.attn(query=x, key=x, value=x, key_padding_mask=1 - mask)
        x = x.transpose(0, 1) 
        return x

class IPALayer(nn.Module):
    def __init__(self, embed_dim, ffn_embed_dim, mha_heads, dropout=0.0,
                 use_rotary_embeddings=False, ipa_args=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.mha_heads = mha_heads
        self.use_rotary_embeddings = use_rotary_embeddings
        self._init_submodules(add_bias_kv=True, dropout=dropout, ipa_args=ipa_args) 

    def _init_submodules(self, add_bias_kv=False, dropout=0.0, ipa_args=None):
        self.adaLN_modulation = nn.Sequential( 
            nn.SiLU(), 
            nn.Linear(self.embed_dim, 6 * self.embed_dim, bias=True)
        )
        self.ipa_norm = nn.LayerNorm(self.embed_dim) 
        self.ipa = InvariantPointAttention(**ipa_args)

        self.mha_l = AttentionWithRoPE(
            self.embed_dim,
            self.mha_heads,
            add_bias_kv=add_bias_kv,
            dropout=dropout,
            use_rotary_embeddings=self.use_rotary_embeddings,
        )

        self.mha_layer_norm = nn.LayerNorm(self.embed_dim, elementwise_affine=False, eps=1e-6) 
        self.fc1 = nn.Linear(self.embed_dim, self.ffn_embed_dim)
        self.fc2 = nn.Linear(self.ffn_embed_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim, elementwise_affine=False, eps=1e-6) 

    def forward(self, x, t, mask=None, frames=None):
        shift_msa_l, scale_msa_l, gate_msa_l, \
            shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t).chunk(6, dim=-1) 
            
        x = x + self.ipa(self.ipa_norm(x), frames, frame_mask=mask)

        residual = x
        x = modulate(self.mha_layer_norm(x), shift_msa_l, scale_msa_l) 
        x = self.mha_l(x, mask=mask)
        x = residual + gate_msa_l.unsqueeze(1) * x

        residual = x
        x = modulate(self.final_layer_norm(x), shift_mlp, scale_mlp)
        x = self.fc2(gelu(self.fc1(x)))
        x = residual + gate_mlp.unsqueeze(1) * x

        return x

class LatentMDGenLayer(nn.Module):
    def __init__(self, embed_dim, ffn_embed_dim, mha_heads, dropout=0.0, num_frames=50, hyena=False,
                 use_rotary_embeddings_t=True, use_rotary_embeddings_l=False, use_time_attention=True, ipa_args=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        self.hyena = hyena 
        self.ffn_embed_dim = ffn_embed_dim
        self.mha_heads = mha_heads
        self.use_time_attention = use_time_attention
        self.use_rotary_embeddings_t = use_rotary_embeddings_t
        self.use_rotary_embeddings_l = use_rotary_embeddings_l
        self._init_submodules(add_bias_kv=True, dropout=dropout, ipa_args=ipa_args) 

    def _init_submodules(self, add_bias_kv=False, dropout=0.0, ipa_args=None):
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.embed_dim, 9 * self.embed_dim, bias=True)
        )

        if ipa_args is not None:
            self.ipa_norm = nn.LayerNorm(self.embed_dim) 
            self.ipa = InvariantPointAttention(**ipa_args)

        if self.hyena:
            self.mha_t = HyenaOperator(
                d_model=self.embed_dim,
                l_max=self.num_frames,
                order=2,
                filter_order=64,
            )
        else:
            self.mha_t = AttentionWithRoPE(
                self.embed_dim,
                self.mha_heads,
                add_bias_kv=add_bias_kv,
                dropout=dropout,
                use_rotary_embeddings=self.use_rotary_embeddings_t,
            )

        self.mha_l = AttentionWithRoPE(
            self.embed_dim,
            self.mha_heads,
            add_bias_kv=add_bias_kv,
            dropout=dropout,
            use_rotary_embeddings=self.use_rotary_embeddings_l,
        )

        self.mha_layer_norm = nn.LayerNorm(self.embed_dim, elementwise_affine=False, eps=1e-6)
        self.fc1 = nn.Linear(self.embed_dim, self.ffn_embed_dim)
        self.fc2 = nn.Linear(self.ffn_embed_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, t, mask=None, frames=None):
        B, T, L, C = x.shape

        shift_msa_l, scale_msa_l, gate_msa_l, \
            shift_msa_t, scale_msa_t, gate_msa_t, \
            shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t).chunk(9, dim=-1)

        # if hasattr(self, 'ipa'): 
        #     x = x + self.ipa(self.ipa_norm(x), frames[:, None], frame_mask=mask)

        residual = x
        x = modulate(self.mha_layer_norm(x), shift_msa_l, scale_msa_l)
        x = self.mha_l(
            x.reshape(B * T, L, C),
            mask=mask.reshape(B * T, L),  
        ).reshape(B, T, L, C)
        x = residual + gate_msa_l.unsqueeze(1) * x

        residual = x
        x = modulate(self.mha_layer_norm(x), shift_msa_t, scale_msa_t)
        if self.hyena:
            assert (mask - 1).sum() == 0
            x = self.mha_t(
                x.transpose(1, 2).reshape(B * L, T, C)
            ).reshape(B, L, T, C).transpose(1, 2)
        else:
            x = self.mha_t(
                x.transpose(1, 2).reshape(B * L, T, C),
                mask=mask.transpose(1, 2).reshape(B * L, T)
            ).reshape(B, L, T, C).transpose(1, 2)
        x = residual + gate_msa_t.unsqueeze(1) * x

        residual = x
        x = modulate(self.final_layer_norm(x), shift_mlp, scale_mlp)
        x = self.fc2(gelu(self.fc1(x)))
        x = residual + gate_mlp.unsqueeze(1) * x

        return x