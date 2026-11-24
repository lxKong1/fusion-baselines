"""
this extremely minimal Decision Transformer model is based on
the following causal transformer (GPT) implementation:

Misha Laskin's tweet:
https://twitter.com/MishaLaskin/status/1481767788775628801?cxt=HHwWgoCzmYD9pZApAAAA

and its corresponding notebook:
https://colab.research.google.com/drive/1NUBqyboDcGte5qAJKOl8gaJC28V_73Iv?usp=sharing

** the above colab notebook has a bug while applying masked_fill
which is fixed in the following code
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
class BevEncoder(nn.Module):
    """Small trainable encoder for MetaDrive's semantic multi-channel BEV.

    Offline frames are stored as uint8 to keep the replay dataset compact.
    Online observations are already float32 in [0, 1].  Normalization is kept
    here so both sources enter exactly the same convolutional network.
    """

    def __init__(self, channels, hidden_dim):
        super().__init__()
        self.channels = int(channels)
        self.network = nn.Sequential(
            nn.Conv2d(self.channels, 32, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, hidden_dim),
        )

    def forward(self, bev):
        if bev.ndim != 5:
            raise ValueError(
                "BEV must have shape [batch, time, channels, height, width], "
                "got {}".format(tuple(bev.shape))
            )
        if bev.shape[2] != self.channels:
            raise ValueError(
                "Expected {} BEV channels, got {}".format(
                    self.channels, bev.shape[2]
                )
            )
        batch, time = bev.shape[:2]
        if bev.dtype == torch.uint8:
            bev = bev.float().div_(255.0)
        else:
            bev = bev.float()
        if not torch.isfinite(bev).all():
            raise ValueError("BEV input contains non-finite values")
        # Small replay/quantization excursions are harmless, but prevent them
        # from changing the online/offline normalization contract.
        bev = bev.clamp(0.0, 1.0)
        encoded = self.network(bev.reshape(batch * time, *bev.shape[2:]))
        return encoded.reshape(batch, time, -1)


class MaskedCausalAttention(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p):
        super().__init__()

        self.n_heads = n_heads
        self.max_T = max_T

        self.q_net = nn.Linear(h_dim, h_dim)
        self.k_net = nn.Linear(h_dim, h_dim)
        self.v_net = nn.Linear(h_dim, h_dim)

        self.proj_net = nn.Linear(h_dim, h_dim)

        self.att_drop = nn.Dropout(drop_p)
        self.proj_drop = nn.Dropout(drop_p)

        ones = torch.ones((max_T, max_T))
        mask = torch.tril(ones).view(1, 1, max_T, max_T)

        # register buffer makes sure mask does not get updated
        # during backpropagation
        self.register_buffer('mask',mask)

    def forward(self, x):
        B, T, C = x.shape # batch size, seq length, h_dim * n_heads

        N, D = self.n_heads, C // self.n_heads # N = num heads, D = attention dim

        # rearrange q, k, v as (B, N, T, D)
        q = self.q_net(x).view(B, T, N, D).transpose(1,2)
        k = self.k_net(x).view(B, T, N, D).transpose(1,2)
        v = self.v_net(x).view(B, T, N, D).transpose(1,2)

        # weights (B, N, T, T)
        weights = q @ k.transpose(2,3) / math.sqrt(D)
        # causal mask applied to weights
        weights = weights.masked_fill(self.mask[...,:T,:T] == 0, float('-inf'))
        # normalize weights, all -inf -> 0 after softmax
        normalized_weights = F.softmax(weights, dim=-1)

        # attention (B, N, T, D)
        attention = self.att_drop(normalized_weights @ v)

        # gather heads and project (B, N, T, D) -> (B, T, N*D)
        attention = attention.transpose(1, 2).contiguous().view(B,T,N*D)

        out = self.proj_drop(self.proj_net(attention))
        return out, normalized_weights


class Block(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p):
        super().__init__()
        self.attention = MaskedCausalAttention(h_dim, max_T, n_heads, drop_p)
        self.mlp = nn.Sequential(
                nn.Linear(h_dim, 4*h_dim),
                nn.GELU(),
                nn.Linear(4*h_dim, h_dim),
                nn.Dropout(drop_p),
            )
        self.ln1 = nn.LayerNorm(h_dim)
        self.ln2 = nn.LayerNorm(h_dim)
        self.count = 0
    def forward(self, x):
        # Attention -> LayerNorm -> MLP -> LayerNorm
        x = x + self.attention(x)[0] # residual
        x = self.ln1(x)
        x = x + self.mlp(x) # residual
        x = self.ln2(x)
        return x

class Fusion_Structure(nn.Module):
    def __init__(self, state_dim, act_dim, n_blocks, h_dim, context_len,
                 n_heads, drop_p, max_timestep=4096, lidar_dim=240,
                 use_bev=False, bev_shape=(5, 84, 84)):
        super().__init__()

        self.state_dim = state_dim
        self.lidar_dim = int(lidar_dim)
        self.act_dim = act_dim
        self.h_dim = h_dim
        self.use_bev = bool(use_bev)
        self.bev_shape = tuple(int(value) for value in bev_shape)
        if len(self.bev_shape) != 3 or min(self.bev_shape) < 1:
            raise ValueError("bev_shape must be (channels, height, width)")
        if self.use_bev:
            self.register_buffer(
                "_bev_input_shape",
                torch.as_tensor(self.bev_shape, dtype=torch.long),
            )

        ### transformer blocks
        input_seq_len = 4 * context_len
        blocks = [Block(h_dim, input_seq_len, n_heads, drop_p) for _ in range(n_blocks)]
        self.transformer = nn.Sequential(*blocks)

        ### projection heads (project to embedding)
        self.embed_ln = nn.LayerNorm(h_dim)
        self.embed_timestep = nn.Embedding(max_timestep, h_dim)
        self.embed_rtg = torch.nn.Linear(1, h_dim)
        self.embed_ctg = torch.nn.Linear(1, h_dim)
        self.embed_lidar = nn.Linear(self.lidar_dim, h_dim)
        if self.use_bev:
            self.embed_bev = BevEncoder(self.bev_shape[0], h_dim)
        aggregate_branches = 3 if self.use_bev else 2
        self.embed_agg = nn.Sequential(*[nn.Linear(h_dim*aggregate_branches, h_dim),
                                        nn.ReLU(),
                                        nn.Linear(h_dim, h_dim)])

        self.embed_state = torch.nn.Linear(state_dim, h_dim)

        # continuous actions
        self.embed_action = torch.nn.Linear(act_dim, h_dim)
        use_action_tanh = True # True for continuous actions

        ### prediction heads
        self.predict_rtg = torch.nn.Linear(h_dim, 1)
        self.predict_ctg = torch.nn.Linear(h_dim, 1)

        self.predict_state = torch.nn.Linear(h_dim, state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(h_dim, act_dim)] + ([nn.Tanh()] if use_action_tanh else []))
        )


    def forward(
        self,
        timesteps,
        states,
        actions,
        returns_to_go,
        returns_to_go_cost,
    ):
        """Run FUSION with the checkpoint's explicit observation modalities."""
        expected_branches = 3 if self.use_bev else 2
        if not isinstance(states, (tuple, list)) or len(states) != expected_branches:
            raise ValueError(
                "Fusion_Structure expects exactly {}, got {} "
                "input branch(es)".format(
                    "[state, lidar, BEV]" if self.use_bev else "[state, lidar]",
                    len(states) if isinstance(states, (tuple, list)) else 0
                )
            )
        state, lidar = states[:2]
        B, T, _ = state.shape
        if lidar.shape[:2] != (B, T):
            raise ValueError(
                "state and lidar batch/time dimensions must match, got {} "
                "and {}".format(tuple(state.shape), tuple(lidar.shape))
            )
        if state.shape[-1] != self.state_dim:
            raise ValueError(
                "Expected state dimension {}, got {}".format(
                    self.state_dim, state.shape[-1]
                )
            )
        if lidar.shape[-1] != self.lidar_dim:
            raise ValueError(
                "Expected lidar dimension {}, got {}".format(
                    self.lidar_dim, lidar.shape[-1]
                )
            )
        if state.isnan().any():
            state = torch.nan_to_num(state, nan=0.0)
            print('replace nan values.')
        if lidar.isnan().any():
            lidar = torch.nan_to_num(lidar, nan=0.0)
            print('replace lidar nan values.')

        time_embeddings = self.embed_timestep(timesteps)
        # # time embeddings are treated similar to positional embeddings
        ego_embeddings = self.embed_state(state)
        lidar_embeddings = self.embed_lidar(lidar)
        observation_embeddings = [ego_embeddings, lidar_embeddings]
        if self.use_bev:
            bev = states[2]
            if tuple(bev.shape[:2]) != (B, T):
                raise ValueError(
                    "state and BEV batch/time dimensions must match, got {} "
                    "and {}".format(tuple(state.shape), tuple(bev.shape))
                )
            if tuple(bev.shape[2:]) != self.bev_shape:
                raise ValueError(
                    "Expected BEV shape {}, got {}".format(
                        self.bev_shape, tuple(bev.shape[2:])
                    )
                )
            observation_embeddings.append(self.embed_bev(bev))
        state_embeddings = self.embed_agg(
            torch.cat(observation_embeddings, dim=2)
        ) + time_embeddings

        action_embeddings = self.embed_action(actions) + time_embeddings
        returns_embeddings = self.embed_rtg(returns_to_go) + time_embeddings
        returns_embeddings_cost = self.embed_ctg(returns_to_go_cost) + time_embeddings

        # stack rtg, states and actions and reshape sequence as
        # (c_0, r_0, s_0, a_0, c_1, r_1, s_1, a_1, c_2, r_2, s_2, a_2 ...)
        h = torch.stack(
            (returns_embeddings_cost, returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(B, 4 * T, self.h_dim)

        h = self.embed_ln(h)
        # print(h.shape)
        # transformer and prediction
        h = self.transformer(h)

        # get h reshaped such that its size = (B x 4 x T x h_dim)
        # that is, for each timestep (t) we have 3 output embeddings from the transformer,
        # each conditioned on all previous timesteps plus
        h = h.reshape(B, T, 4, self.h_dim).permute(0, 2, 1, 3)

        # get predictions conditioned on the attended "causal parents"
        return_cost_preds = self.predict_ctg(h[:,3])     # predict next rtg given c, r, s, a
        return_preds = self.predict_rtg(h[:,3])     # predict next rtg given c, r, s, a
        state_preds = self.predict_state(h[:,3])    # predict next state given c, r, s, a
        action_preds = self.predict_action(h[:,2])  # predict action given c, r, s

        return state_preds, action_preds, return_preds, return_cost_preds
