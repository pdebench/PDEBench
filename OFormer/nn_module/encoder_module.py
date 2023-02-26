import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
from torch.nn.init import xavier_uniform_, constant_, xavier_normal_
from .attention_module import pair, PreNorm, PostNorm,\
    StandardAttention, FeedForward, LinearAttention, ReLUFeedForward
from .cnn_module import PeriodicConv2d, PeriodicConv3d, UpBlock
#from .gnn_module import SmoothConvEncoder, SmoothConvDecoder, index_points
#from torch_scatter import scatter
# helpers


class Transformer(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 heads,
                 dim_head,
                 attn_type,               # ['standard', 'galerkin', 'fourier']
                 mlp_dim, dropout=0.):
        super().__init__()
        assert attn_type in ['standard', 'galerkin', 'fourier']
        self.layers = nn.ModuleList([])

        if attn_type == 'standard':
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    PreNorm(dim, StandardAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
                ]))
        else:

            for _ in range(depth):
                if attn_type == 'galerkin':
                    attn_module = GalerkinAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
                else:           # attn_type == 'fourier'
                    attn_module = FourierAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)

                attn_module._init_params()

                self.layers.append(nn.ModuleList([
                    PreNorm(dim, attn_module),
                    FeedForward(dim, mlp_dim, dropout=dropout)
                ]))

    def forward(self, x, pos_embedding=None):
        for attn, ff in self.layers:
            if pos_embedding is not None:
                x = x + pos_embedding
            x = attn(x) + x
            x = ff(x) + x
        return x


class STTransformer(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 heads,
                 dim_head,
                 mlp_dim,
                 attn_type,  # ['standard', 'galerkin', 'fourier']
                 dropout=0.):
        super().__init__()
        assert attn_type in ['standard', 'galerkin', 'fourier']
        self.layers = nn.ModuleList([])
        self.attn_type = attn_type
        if attn_type == 'standard':
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    # spatial
                    PreNorm(dim, StandardAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    # temporal
                    PreNorm(dim, StandardAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                ]))
        else:

            for _ in range(depth):
                if attn_type == 'galerkin':
                    attn_module1 = GalerkinAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
                    attn_module2 = GalerkinAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)

                else:  # attn_type == 'fourier'
                    attn_module1 = FourierAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
                    attn_module2 = FourierAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)

                self.layers.append(nn.ModuleList([
                    PreNorm(dim, attn_module1),
                    FeedForward(dim, mlp_dim, dropout=dropout),
                    PreNorm(dim, attn_module2),
                    FeedForward(dim, mlp_dim, dropout=dropout),
                ]))

    def forward(self, x, temp_embedding, pos_embedding):
        b, c, t, h, w = x.shape
        for layer_no, (spa_attn, ff1, temp_attn, ff2) in enumerate(self.layers):
            if layer_no == 0:
                x = rearrange(x, 'b c t h w -> (b t) (h w) c')
            else:
                x = rearrange(x, '(b h w) t c -> (b t) (h w) c', h=h, w=w)
            x = x + pos_embedding

            x = spa_attn(x) + x
            x = x + pos_embedding
            x = ff1(x) + x

            x = rearrange(x, '(b t) (h w) c -> (b h w) t c', t=t, h=h)

            x = x + temp_embedding
            x = temp_attn(x) + x
            x = x + temp_embedding
            x = ff2(x) + x

            if layer_no == len(self.layers) - 1:
                x = rearrange(x, '(b h w) t c -> b c t h w', h=h, w=w)

        return x

    def forward_with_clstoken(self, x, x_cls, temp_embedding, pos_embedding):
        b, c, t, h, w = x.shape
        for layer_no, (spa_attn, ff1, temp_attn, ff2) in enumerate(self.layers):
            if layer_no == 0:
                x = rearrange(x, 'b c t h w -> (b t) (h w) c')
            else:
                x = rearrange(x, '(b h w) t c -> (b t) (h w) c', h=h, w=w)

            x = x + pos_embedding
            x = spa_attn(x) + x
            x = ff1(x) + x

            x = rearrange(x, '(b t) (h w) c -> (b h w) t c', t=t, h=h)
            if layer_no == 0:
                x_cls = repeat(x_cls, '() n d -> b n d', b=x.shape[0])
            x = torch.cat([x_cls, x], dim=1)  # [bhw, t+1, d]

            x = x + temp_embedding
            x = temp_attn(x) + x

            x = ff2(x) + x
            x = x[:, 1:]
            x_cls = x[:, 0:1]
            if layer_no == len(self.layers) - 1:
                x = rearrange(x, '(b h w) t c -> b c t h w', h=h, w=w)
                x_cls = rearrange(x_cls, '(b h w) t c -> b c (t h w)', h=h, w=w)  # here t=1
                x_cls = x_cls.mean(dim=-1)
        return x, x_cls


class STTransformerCat(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 heads,
                 dim_head,
                 mlp_dim,
                 attn_type,  # ['standard', 'galerkin', 'fourier']
                 dropout=0.,
                 attention_init='xavier'):
        super().__init__()
        assert attn_type in ['standard', 'galerkin', 'fourier']
        self.layers = nn.ModuleList([])
        self.attn_type = attn_type
        if attn_type == 'standard':
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    # spatial
                    nn.ModuleList([
                    nn.LayerNorm(dim),
                    StandardAttention(dim+2, heads=heads, dim_head=dim_head, dropout=dropout),
                    nn.Linear(dim+2, dim),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))]),
                    # temporal
                    nn.ModuleList([
                    nn.LayerNorm(dim),
                    StandardAttention(dim+1, heads=heads, dim_head=dim_head, dropout=dropout),
                    nn.Linear(dim + 2, dim),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))]),
                ]))
        else:

            for d in range(depth):
                if attn_type == 'galerkin':
                    attn_module1 = GalerkinAttention(dim+2, heads=heads, dim_head=dim_head, dropout=dropout,
                                                     relative_emb=True, scale=32/(4**d), init_method=attention_init)
                    attn_module2 = GalerkinAttention1D(dim+1, heads=heads, dim_head=dim_head, dropout=dropout,
                                                       relative_emb=True, init_method=attention_init)

                else:  # attn_type == 'fourier'
                    attn_module1 = FourierAttention(dim+2, heads=heads, dim_head=dim_head, dropout=dropout)
                    attn_module2 = FourierAttention(dim+1, heads=heads, dim_head=dim_head, dropout=dropout)

                self.layers.append(nn.ModuleList([
                    # spatial
                    nn.ModuleList([
                    nn.LayerNorm(dim),
                    attn_module1,
                    nn.Linear(dim+2, dim),
                    FeedForward(dim, mlp_dim, dropout=dropout)]),

                    # temporal
                    nn.ModuleList([
                    nn.LayerNorm(dim),
                    attn_module2,
                    nn.Linear(dim + 1, dim),
                    FeedForward(dim, mlp_dim, dropout=dropout)]),
                ]))

    def forward(self, x, x_cls, temp_embedding, pos_embedding):
        # x in [b t n c]
        b, t, n, c = x.shape
        pos_embedding = repeat(pos_embedding, 'b n c -> (b repeat) n c', repeat=t)  # [b*t, n, c]

        temp_embedding = repeat(temp_embedding, '() t c -> b t c', b=b*n)
        for layer_no, (spa_attn, temp_attn) in enumerate(self.layers):

            if layer_no == 0:
                x = rearrange(x, 'b t n c -> (b t) n c')
            else:
                x = rearrange(x, '(b n) t c -> (b t) n c', n=n)

            # spatial attention
            [ln, attn, proj, ffn] = spa_attn

            x = ln(x)
            x = torch.cat((x, pos_embedding), dim=-1)  # [b, n, c+2]
            x = attn(x, pos_embedding) + x
            x = proj(x)
            x = ffn(x) + x

            # temporal attention
            x = rearrange(x, '(b t) n c -> (b n) t c', t=t)
            if layer_no == 0:
                x_cls = repeat(x_cls, '() n d -> b n d', b=x.shape[0])  # n=1
            [ln, attn, proj, ffn] = temp_attn

            x = ln(x)
            x = torch.cat([x_cls, x], dim=1)  # [bhw, t+1, c]
            x = torch.cat((x, temp_embedding), dim=-1)  # [b, n, c+1]
            x = attn(x) + x
            x = proj(x)
            x = ffn(x) + x

            x = x[:, 1:]
            x_cls = x[:, 0:1]
            if layer_no == len(self.layers) - 1:
                x = rearrange(x, '(b n) t c -> b n t c', n=n)
                x_cls = rearrange(x_cls, '(b n) t c -> b c (t n)', n=n)  # here t=1, n=number of pivotal points
                x_cls = x_cls.mean(dim=-1)
        return x, x_cls


class STTransformerCatNoCls(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 heads,
                 dim_head,
                 mlp_dim,
                 attn_type,  # ['standard', 'galerkin', 'fourier']
                 use_ln=False,
                 scale=32,  # can be list, or an int
                 dropout=0.,
                 relative_emb_dim=2,
                 min_freq=1/64,
                 attention_init='orthogonal'):
        super().__init__()
        assert attn_type in ['standard', 'galerkin', 'fourier']

        if isinstance(scale, int):
            scale = [scale] * depth

        self.layers = nn.ModuleList([])
        self.attn_type = attn_type
        self.use_ln = use_ln

        if attn_type == 'standard':
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    # spatial
                    nn.ModuleList([
                    PreNorm(dim, StandardAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))]),
                    # temporal
                    nn.ModuleList([
                    PreNorm(dim, StandardAttention(dim+1, heads=heads, dim_head=dim_head, dropout=dropout)),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))]),
                ]))
        else:

            for d in range(depth):
                # spatial
                attn_module1 = LinearAttention(dim, attn_type,
                                               heads=heads, dim_head=dim_head, dropout=dropout,
                                               relative_emb=True, scale=scale[d],
                                               relative_emb_dim=relative_emb_dim,
                                               min_freq=min_freq,
                                               init_method=attention_init
                                                )
                # temporal
                attn_module2 = LinearAttention(dim, attn_type,
                                               heads=heads, dim_head=dim_head, dropout=dropout,
                                               relative_emb=True, scale=1,
                                               relative_emb_dim=1,
                                               min_freq=1,
                                               init_method=attention_init)
                if not use_ln:

                    self.layers.append(nn.ModuleList([
                        # spatial
                        nn.ModuleList([
                        attn_module1,
                        FeedForward(dim, mlp_dim, dropout=dropout)]),

                        # temporal
                        nn.ModuleList([
                        attn_module2,
                        FeedForward(dim, mlp_dim, dropout=dropout)]),
                    ]))
                else:
                    self.layers.append(nn.ModuleList([
                        # spatial
                        nn.ModuleList([
                            nn.LayerNorm(dim),
                            attn_module1,
                            FeedForward(dim, mlp_dim, dropout=dropout),
                        ]),

                        # temporal
                        nn.ModuleList([
                            nn.LayerNorm(dim),
                            attn_module2,
                            FeedForward(dim, mlp_dim, dropout=dropout),
                        ]),
                    ]))

    def forward(self, x, pos_embedding):
        # x in [b t n c]
        b, t, n, c = x.shape
        pos_embedding = repeat(pos_embedding, 'b n c -> (b repeat) n c', repeat=t)  # [b*t, n, c]
        temp_embedding = torch.arange(t).float().to(x.device).view(1, t, 1)
        temp_embedding = repeat(temp_embedding, '() t c -> b t c', b=b*n)

        for layer_no, (spa_attn, temp_attn) in enumerate(self.layers):
            if layer_no == 0:
                x = rearrange(x, 'b t n c -> (b t) n c')
            else:
                x = rearrange(x, '(b n) t c -> (b t) n c', n=n)

            # spatial attention
            if not self.use_ln:
                [attn, ffn] = spa_attn

                x = attn(x, pos_embedding) + x
                x = ffn(x) + x

            else:
                [ln, attn, ffn] = spa_attn
                x = ln(x)
                x = attn(x, pos_embedding) + x
                x = ffn(x) + x

            # temporal attention
            x = rearrange(x, '(b t) n c -> (b n) t c', t=t)

            if not self.use_ln:
                [attn, ffn] = temp_attn

                x = attn(x, temp_embedding) + x
                x = ffn(x) + x
            else:
                [ln, attn, ffn] = temp_attn
                x = ln(x)
                x = attn(x, temp_embedding, not_assoc=True) + x
                x = ffn(x) + x

            if layer_no == len(self.layers) - 1:
                x = rearrange(x, '(b n) t c -> b n t c', n=n)
        return x


class TransformerCat(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 heads,
                 dim_head,
                 mlp_dim,
                 attn_type,  # ['standard', 'galerkin', 'fourier']
                 scale=16,
                 dropout=0.,
                 attention_init='xavier'):
        super().__init__()
        assert attn_type in ['standard', 'galerkin', 'fourier']

        if isinstance(scale, int):
            scale = [scale] * depth

        self.layers = nn.ModuleList([])
        self.attn_type = attn_type
        if attn_type == 'standard':
            for _ in range(depth):
                self.layers.append(
                    nn.ModuleList([
                    nn.LayerNorm(dim),
                    StandardAttention(dim+2, heads=heads, dim_head=dim_head, dropout=dropout),
                    nn.Linear(dim+2, dim),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))]),
                )
        else:

            for d in range(depth):
                if attn_type == 'galerkin':
                    attn_module1 = GalerkinAttention(dim+2, heads=heads, dim_head=dim_head, dropout=dropout,
                                                     relative_emb=True, scale=scale[d], init_method=attention_init)

                else:  # attn_type == 'fourier'
                    attn_module1 = FourierAttention(dim+2, heads=heads, dim_head=dim_head, dropout=dropout)

                self.layers.append(
                    # spatial
                    nn.ModuleList([
                    nn.LayerNorm(dim),
                    attn_module1,
                    nn.Linear(dim+2, dim),
                    FeedForward(dim, mlp_dim, dropout=dropout)]),
                    )

    def forward(self, x, x_cls, pos_embedding, cls_embedding):
        # x in [b n c], pos_embedding in [b n 2], x_cls in [b c], cls_emb in [b 1 2]
        b, n, c = x.shape
        x_cls = rearrange(x_cls, 'b c -> b 1 c')
        if x_cls.shape[0] != b:
            x_cls = repeat(x_cls, '1 1 c -> b 1 c', b=b)
        cls_embedding = repeat(cls_embedding, '() 1 c -> b 1 c', b=b)
        pos_embedding = torch.cat((pos_embedding,
                                   cls_embedding), dim=1)
        x = torch.cat((x_cls, x), dim=1)  # [b, n+1, c]
        for layer_no, attn in enumerate(self.layers):

            [ln, attn, proj, ffn] = attn

            x = ln(x)
            x = torch.cat((x, pos_embedding), dim=-1)  # [b, n, c+2]
            x = attn(x, pos_embedding) + x
            x = proj(x)
            x = ffn(x) + x

        x = x[:, 1:]
        x_cls = x[:, 0:1]
        return x, x_cls


class TransformerCatNoCls(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 heads,
                 dim_head,
                 mlp_dim,
                 attn_type,  # ['standard', 'galerkin', 'fourier']
                 use_ln=False,
                 scale=16,     # can be list, or an int
                 dropout=0.,
                 relative_emb_dim=2,
                 min_freq=1/64,
                 attention_init='orthogonal',
                 init_gain=None,
                 use_relu=False,
                 cat_pos=False,
                 ):
        super().__init__()
        assert attn_type in ['standard', 'galerkin', 'fourier']

        if isinstance(scale, int):
            scale = [scale] * depth
        assert len(scale) == depth

        self.layers = nn.ModuleList([])
        self.attn_type = attn_type
        self.use_ln = use_ln

        if attn_type == 'standard':
            for _ in range(depth):
                self.layers.append(
                    nn.ModuleList([
                    PreNorm(dim, StandardAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                    PreNorm(dim,  FeedForward(dim, mlp_dim, dropout=dropout)
                                  if not use_relu else ReLUFeedForward(dim, mlp_dim, dropout=dropout))]),
                )
        else:
            for d in range(depth):
                if scale[d] != -1 or cat_pos:
                    attn_module = LinearAttention(dim, attn_type,
                                                   heads=heads, dim_head=dim_head, dropout=dropout,
                                                   relative_emb=True, scale=scale[d],
                                                   relative_emb_dim=relative_emb_dim,
                                                   min_freq=min_freq,
                                                   init_method=attention_init,
                                                   init_gain=init_gain
                                                   )
                else:
                    attn_module = LinearAttention(dim, attn_type,
                                                  heads=heads, dim_head=dim_head, dropout=dropout,
                                                  cat_pos=True,
                                                  pos_dim=relative_emb_dim,
                                                  relative_emb=False,
                                                  init_method=attention_init,
                                                  init_gain=init_gain
                                                  )
                if not use_ln:
                    self.layers.append(
                        nn.ModuleList([
                                        attn_module,
                                        FeedForward(dim, mlp_dim, dropout=dropout)
                                        if not use_relu else ReLUFeedForward(dim, mlp_dim, dropout=dropout)
                        ]),
                        )
                else:
                    self.layers.append(
                        nn.ModuleList([
                            nn.LayerNorm(dim),
                            attn_module,
                            nn.LayerNorm(dim),
                            FeedForward(dim, mlp_dim, dropout=dropout)
                            if not use_relu else ReLUFeedForward(dim, mlp_dim, dropout=dropout),
                        ]),
                    )

    def forward(self, x, pos_embedding):
        # x in [b n c], pos_embedding in [b n 2]
        b, n, c = x.shape

        for layer_no, attn_layer in enumerate(self.layers):
            if not self.use_ln:
                [attn, ffn] = attn_layer

                x = attn(x, pos_embedding) + x
                x = ffn(x) + x
            else:
                [ln1, attn, ln2, ffn] = attn_layer
                x = ln1(x)
                x = attn(x, pos_embedding) + x
                x = ln2(x)
                x = ffn(x) + x
        return x

class LocalTransformer(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 heads,
                 dim_head,
                 mlp_dim,
                 attn_type,  # ['standard', 'galerkin', 'fourier']
                 dropout=0.):
        super().__init__()
        assert attn_type in ['standard', 'galerkin', 'fourier']
        self.layers = nn.ModuleList([])
        self.attn_type = attn_type
        if attn_type == 'standard':
            for _ in range(depth):
                self.layers.append(
                    # spatial
                    nn.ModuleList([
                        nn.LayerNorm(dim),
                        StandardAttention(dim + 2, heads=heads, dim_head=dim_head, dropout=dropout),
                        nn.Linear(dim + 2, dim),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))]),
                    )
        else:

            for _ in range(depth):
                if attn_type == 'galerkin':
                    attn_module1 = GalerkinAttention(dim + 2, heads=heads, dim_head=dim_head, dropout=dropout)

                else:  # attn_type == 'fourier'
                    attn_module1 = FourierAttention(dim + 2, heads=heads, dim_head=dim_head, dropout=dropout)

                self.layers.append(
                    nn.ModuleList([
                        nn.LayerNorm(dim),
                        attn_module1,
                        nn.Linear(dim + 2, dim),
                        FeedForward(dim, mlp_dim, dropout=dropout)]),
                )

    def forward(self, x, pos_embedding):
        # x in [b, t, p, n, c]
        # pos_embedding in [b, p, n, 2]
        b, t, p, n, c = x.shape   # p: num of patches, n: num of points inside each patch (padded)

        pos_embedding = rearrange(
            repeat(pos_embedding, 'b p n c -> (b repeat) p n c', repeat=t),
            'bt p n c -> (bt p) n c')                       # [b*t*p, n, c]

        x = rearrange(x, 'b t p n c -> (b t p) n c')

        for layer_no, [ln, attn, proj, ffn] in enumerate(self.layers):

            x = ln(x)
            x = torch.cat((x, pos_embedding), dim=-1)  # [b, n, c+2]
            x = attn(x) + x
            x = proj(x)
            x = ffn(x) + x

        x = rearrange(x, '(b t p) n c -> b t p n c', b=b, t=t, p=p)

        return x


class CNNEncoder(nn.Module):
    def __init__(self,
                 input_channels,           # how many channels
                 in_grid_size,             # this should be the input image height/width
                 seq_len,                  # this should be the input sequence length
                 in_emb_dim,               # embedding dim of token                 (how about 512)
                 out_seq_emb_dim,          # embedding dim of encoded sequence      (how about 256)
                 depth,                    # depth of transformer / how many layers of attention    (4)
                 emb_dropout=0.1,           # dropout of embedding
                 ):
        super().__init__()
        h, w = pair(in_grid_size // 4)
        t = seq_len
        self.in_grid_size = in_grid_size

        self.to_embedding = nn.Sequential(
            PeriodicConv3d(input_channels, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                           spatial_pad=1, temp_pad=1, bias=False),
            nn.GELU(),
            PeriodicConv3d(64, in_emb_dim, kernel_size=(1, 3, 3), stride=(1, 2, 2),
                           spatial_pad=1, temp_pad=0, bias=False),    # [t, h/2, w/2]
            nn.GELU(),
            PeriodicConv3d(in_emb_dim, in_emb_dim, kernel_size=(1, 3, 3), stride=(1, 2, 2),
                           spatial_pad=1, temp_pad=0, bias=False),   # [t, h/4, w/4]
        )
        self.dropout = nn.Dropout(emb_dropout)

        self.net = nn.ModuleList([nn.Sequential(
                                PeriodicConv3d(in_emb_dim, in_emb_dim//4, kernel_size=(1, 1, 1), stride=(1, 1, 1),
                                               spatial_pad=0, temp_pad=0, bias=False),
                                nn.GELU(),
                                PeriodicConv3d(in_emb_dim//4, in_emb_dim//4, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                                               spatial_pad=1, temp_pad=1, bias=False),
                                nn.GELU(),
                                PeriodicConv3d(in_emb_dim//4, in_emb_dim, kernel_size=(1, 1, 1), stride=(1, 1, 1),
                                               spatial_pad=0, temp_pad=0, bias=False),)
                                for _ in range(depth) ])

        # squeeze the temporal dimension
        self.to_init1 = nn.Sequential(
            nn.Conv3d(in_emb_dim, in_emb_dim,
                      kernel_size=(t, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=False),
            nn.GELU()
        )

        self.to_init2 = nn.Sequential(
            nn.Conv2d(in_emb_dim, in_emb_dim, kernel_size=1, stride=1, padding=0, bias=False))

        # upsample the space resolution and go back to the original resolution
        self.up_block_num = 2
        self.up_layers = []
        for _ in range(self.up_block_num):
            self.up_layers.append(UpBlock(in_emb_dim, in_emb_dim))
        self.up_layers = nn.Sequential(*self.up_layers)

        self.to_out = nn.Sequential(
            nn.Conv2d(in_emb_dim, in_emb_dim, 1, 1, 0, bias=False),
            nn.GELU(),
            nn.Conv2d(in_emb_dim, in_emb_dim, 1, 1, 0, bias=False),
            nn.GELU(),
            nn.Conv2d(in_emb_dim, out_seq_emb_dim, 1, 1, 0, bias=False),
            nn.InstanceNorm2d(out_seq_emb_dim))

        self.to_cls = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Conv2d(in_emb_dim, in_emb_dim, 4, 4, 0, bias=False),
            nn.GELU(),
            nn.Conv2d(in_emb_dim, out_seq_emb_dim, 1, 1, 0, bias=False),
            Rearrange('b c h w -> b (h w) c'),
            nn.LayerNorm(out_seq_emb_dim))

    def forward(self, x):
        x = self.to_embedding(x)
        x = self.dropout(x)
        for layer in self.net:
            x = layer(x) + x
        x = rearrange(self.to_init1(x), 'b c 1 h w -> b c h w')   # [b, c, h, w]
        x = self.to_init2(x)
        x_cls = self.to_cls(x).view(x.shape[0], -1)
        x = self.up_layers(x)
        x = self.to_out(x)
        return x, x_cls


class GraphEncoder(nn.Module):
    def __init__(self,
                 input_channels,           # how many channels
                 seq_len,                  # this should be the input sequence length
                 in_emb_dim,               # embedding dim of token                 (how about 512)
                 out_seq_emb_dim,          # embedding dim of encoded sequence      (how about 256)
                 depth,                    # depth of transformer / how many layers of attention    (4)
                 emb_dropout=0.1,           # dropout of embedding
                 ):
        super().__init__()

        t = seq_len
        self.dropout = nn.Dropout(emb_dropout)
        self.temp_embedding = nn.Parameter(
            torch.cat((torch.tensor([-1.]), torch.linspace(0, 1, t)), dim=0).view(1, t+1, 1), requires_grad=False)   # [b, t, 1]
        self.cls_token = nn.Parameter(torch.randn(1, 1, in_emb_dim), requires_grad=True)
        self.gamma = nn.Parameter(torch.tensor([0.]), requires_grad=True)
        # define the model
        self.encoder = SmoothConvEncoder(input_channels, in_emb_dim, 3)

        self.transformer = STTransformerCat(in_emb_dim, depth, 8, 64, in_emb_dim, 'galerkin')

        self.to_cls = nn.Sequential(
            nn.Linear(in_emb_dim, out_seq_emb_dim, bias=False),
            nn.LayerNorm(out_seq_emb_dim))

        self.project_to_latent = nn.Sequential(
            nn.InstanceNorm1d(in_emb_dim),
            nn.Linear(in_emb_dim, in_emb_dim, bias=False),
            nn.GELU(),
            nn.Linear(in_emb_dim, in_emb_dim, bias=False),
            nn.GELU(),
            nn.Linear(in_emb_dim, out_seq_emb_dim, bias=False),
            nn.InstanceNorm1d(out_seq_emb_dim))

    def forward(self, x,
                input_pos, pivotal_pos,
                input2input_graph, input2pivot_graph,
                input2input_cutoff, input2pivot_cutoff
                ):
        # x in shape [b, c, t, n]
        # first, we encode the information from input nodes to pivotal nodes
        x = self.encoder.forward(x, input_pos, pivotal_pos, input2input_graph, input2pivot_graph,
                                 input2input_cutoff, input2pivot_cutoff)  # expect x back in [b, c, t, n]
        x = self.dropout(x)
        x, x_cls = self.transformer.forward(x,
                                          self.cls_token,
                                          self.temp_embedding, pivotal_pos)
        # squeeze the temporal embedding
        # x: [b n t c]
        x = torch.sum(x, dim=2)  # [b, n, c]

        x, x_cls = self.project_to_latent(x), self.to_cls(x_cls) * self.gamma
        return x, x_cls


class FullyAttentionEncoder(nn.Module):
    def __init__(self,
                 input_channels,           # how many channels
                 seq_len,                  # this should be the input sequence length
                 in_emb_dim,               # embedding dim of token                 (how about 512)
                 out_seq_emb_dim,          # embedding dim of encoded sequence      (how about 256)
                 depth,                    # depth of transformer / how many layers of attention    (4)
                 n_patch=16,
                 out_grid=64,
                 emb_dropout=0.1,           # dropout of embedding
                 ):
        super().__init__()
        self.n_patch = n_patch
        self.out_grid = out_grid

        t = seq_len
        self.dropout = nn.Dropout(emb_dropout)

        self.to_embedding = nn.Sequential(
            nn.Linear(input_channels, in_emb_dim, bias=False),
            nn.GELU(),
            nn.Linear(in_emb_dim, in_emb_dim, bias=False))

        self.encoding_transformer = LocalTransformer(in_emb_dim, 2, 8, 64, in_emb_dim, 'galerkin')

        self.temp_embedding = nn.Parameter(
            torch.cat((torch.tensor([-1.]), torch.linspace(0, 1, t)), dim=0).view(1, t+1, 1), requires_grad=False)   # [b, t, 1]
        self.cls_token = nn.Parameter(torch.randn(1, 1, in_emb_dim), requires_grad=True)
        self.gamma = nn.Parameter(torch.tensor([0.]), requires_grad=True)

        self.transformer = STTransformerCat(in_emb_dim, depth, 8, 64, in_emb_dim, 'galerkin')
        self.temporal_norm = nn.LayerNorm(t)

        self.to_cls = nn.Sequential(
            nn.Linear(in_emb_dim, out_seq_emb_dim, bias=False),
            nn.LayerNorm(out_seq_emb_dim))

        self.project_to_latent = nn.Sequential(
            nn.Linear(in_emb_dim, in_emb_dim, bias=False),
            nn.InstanceNorm1d(in_emb_dim))

        # upsample the space resolution and go back to the original resolution
        self.up_block_num = int(np.log2(out_grid//n_patch))
        self.up_layers = []
        for _ in range(self.up_block_num):
            self.up_layers.append(UpBlock(in_emb_dim, in_emb_dim))
        self.up_layers = nn.Sequential(*self.up_layers)

        self.to_out = nn.Sequential(
            nn.Conv2d(in_emb_dim, out_seq_emb_dim, 1, 1, 0, bias=False),
            nn.InstanceNorm2d(out_seq_emb_dim))


    def forward(self,
                x,  # [b, t, p, n_p, c]
                dist2patch_center,  # [b, p, n_p, 2]
                patch_pos,   # [b, p, 2]
                num_of_points_per_patch   # [b, p]
                ):
        b, t = x.shape[0:2]  # num of frames
        x = self.to_embedding(x)
        x = self.dropout(x)
        x = self.encoding_transformer.forward(x, dist2patch_center)    # [b, t, p, n_p, c]

        # performing mean pooling, result in shape: [b, t, p, c]
        x = x.sum(dim=-2) / repeat(num_of_points_per_patch.view((b, 1, -1, 1)), 'b () p 1 -> b repeat p 1', repeat=t)

        x, x_cls = self.transformer.forward(x,
                                            self.cls_token,
                                            self.temp_embedding, patch_pos)
        # squeeze the temporal embedding
        # x: [b n t c]
        x = rearrange(x, 'b n t c -> b n c t')
        x = self.temporal_norm(x)
        x = torch.sum(x, dim=-1)  # [b, n, c]

        x, x_cls = self.project_to_latent(x), self.to_cls(x_cls) * self.gamma

        # x: [b n c]
        x = rearrange(x, 'b (h w) c -> b c h w', h=self.n_patch)
        x = self.up_layers(x)
        x = self.to_out(x)
        return x, x_cls


class PureAttentionEncoder(nn.Module):
    def __init__(self,
                 input_channels,           # how many channels
                 seq_len,                  # this should be the input sequence length
                 in_emb_dim,               # embedding dim of token                 (how about 512)
                 out_seq_emb_dim,          # embedding dim of encoded sequence      (how about 256)
                 depth,                    # depth of transformer / how many layers of attention    (4)
                 n_patch=16,
                 out_grid=64,
                 emb_dropout=0.1,           # dropout of embedding
                 ):
        super().__init__()
        self.n_patch = n_patch
        self.out_grid = out_grid

        t = seq_len
        self.dropout = nn.Dropout(emb_dropout)

        self.to_embedding = nn.Sequential(
            nn.Linear(input_channels, in_emb_dim, bias=False),
            nn.GELU(),
            nn.Linear(in_emb_dim, in_emb_dim, bias=False),
            nn.GELU(),
            nn.Linear(in_emb_dim, in_emb_dim, bias=False),
        )

        self.temp_embedding = nn.Parameter(
            torch.cat((torch.tensor([-1.]), torch.linspace(0, 1, t)), dim=0).view(1, t+1, 1), requires_grad=False)   # [b, t, 1]
        self.cls_token = nn.Parameter(torch.randn(1, 1, in_emb_dim), requires_grad=True)
        self.cls_emb = nn.Parameter(torch.randn(1, 1, 2), requires_grad=True)
        self.gamma = nn.Parameter(torch.tensor([0.1]), requires_grad=True)

        self.st_transformer = STTransformerCat(in_emb_dim, depth-1, 4, 64, in_emb_dim, 'galerkin')

        self.s_transformer = TransformerCat(in_emb_dim*2, 1, 4, 64, in_emb_dim*2, 'galerkin')

        self.to_cls = nn.Sequential(
            nn.Linear(2*in_emb_dim, out_seq_emb_dim, bias=False),
            nn.GELU(),
            nn.Linear(out_seq_emb_dim, out_seq_emb_dim, bias=False),
            nn.GELU(),
            nn.Linear(out_seq_emb_dim, out_seq_emb_dim, bias=False),
            nn.LayerNorm(out_seq_emb_dim))

        self.shrink_temporal = nn.Sequential(
            Rearrange('b n t c -> b n (t c)'),
            nn.Linear(t*in_emb_dim, in_emb_dim, bias=False),
        )
        self.expand_feat = nn.Linear(in_emb_dim, 2*in_emb_dim)

        self.project_to_latent = nn.Sequential(
            nn.Linear(2*in_emb_dim, out_seq_emb_dim, bias=False),
            nn.GELU(),
            nn.Linear(out_seq_emb_dim, out_seq_emb_dim, bias=False),
            nn.GELU(),
            nn.Linear(out_seq_emb_dim, out_seq_emb_dim, bias=False),
            nn.InstanceNorm1d(in_emb_dim))

    def forward(self,
                x,  # [b, c, t, n]
                input_pos,  # [b, n, 2]
                ):
        x = rearrange(x, 'b c t n-> b t n c')
        x = self.to_embedding(x)
        x = self.dropout(x)

        x, x_cls = self.st_transformer.forward(x,
                                             self.cls_token,
                                             self.temp_embedding,
                                             input_pos)
        x = self.shrink_temporal(x)
        x, x_cls = self.expand_feat(x), self.expand_feat(x_cls)
        x, x_cls = self.s_transformer.forward(x, x_cls, input_pos, self.cls_emb)
        x, x_cls = self.project_to_latent(x), self.to_cls(x_cls) * self.gamma

        return x, x_cls


class PoolingAttentionEncoder(nn.Module):
    def __init__(self,
                 input_channels,           # how many channels
                 seq_len,                  # this should be the input sequence length
                 in_emb_dim,               # embedding dim of token                 (how about 512)
                 out_seq_emb_dim,          # embedding dim of encoded sequence      (how about 256)
                 depth,                    # depth of transformer / how many layers of attention    (4)
                 n_patch=16,
                 out_grid=64,
                 emb_dropout=0.1,           # dropout of embedding
                 ):
        super().__init__()
        self.n_patch = n_patch
        self.out_grid = out_grid

        t = seq_len
        self.dropout = nn.Dropout(emb_dropout)

        self.to_embedding = nn.Sequential(
            nn.Linear(input_channels, in_emb_dim, bias=False),
            nn.GELU(),
            nn.Linear(in_emb_dim, in_emb_dim, bias=False),
            nn.GELU(),
            nn.Linear(in_emb_dim, in_emb_dim, bias=False),
            nn.LayerNorm(in_emb_dim)
        )

        self.pooling_layer = AttentivePooling(in_emb_dim, 4, 64)

        self.temp_embedding = nn.Parameter(
            torch.cat((torch.tensor([-1.]), torch.linspace(0, 1, t)), dim=0).view(1, t+1, 1), requires_grad=False)   # [b, t, 1]
        self.cls_token = nn.Parameter(torch.randn(1, 1, in_emb_dim), requires_grad=True)
        self.cls_emb = nn.Parameter(torch.randn(1, 1, 2), requires_grad=True)
        self.gamma = nn.Parameter(torch.tensor([0.1]), requires_grad=True)

        self.st_transformer = STTransformerCat(in_emb_dim, 2, 4, 64, in_emb_dim, 'galerkin')

        self.s_transformer = TransformerCat(in_emb_dim*2, 1, 4, 64, in_emb_dim*2, 'galerkin')

        self.to_cls = nn.Sequential(
            nn.Linear(2*in_emb_dim, out_seq_emb_dim, bias=False),
            nn.GELU(),
            nn.Linear(out_seq_emb_dim, out_seq_emb_dim, bias=False),
            nn.GELU(),
            nn.Linear(out_seq_emb_dim, out_seq_emb_dim, bias=False),
            nn.LayerNorm(out_seq_emb_dim))

        self.shrink_temporal = nn.Sequential(
            Rearrange('b n t c -> b n (t c)'),
            nn.Linear(t*in_emb_dim, in_emb_dim, bias=False),
        )
        self.expand_feat = nn.Linear(in_emb_dim, 2*in_emb_dim)

        self.project_to_latent = nn.Sequential(
            nn.Linear(2*in_emb_dim, out_seq_emb_dim, bias=False),
            nn.GELU(),
            nn.Linear(out_seq_emb_dim, out_seq_emb_dim, bias=False),
            nn.GELU(),
            nn.Linear(out_seq_emb_dim, out_seq_emb_dim, bias=False),
            nn.InstanceNorm1d(in_emb_dim))

    def forward(self,
                x,  # [b, c, t, n]
                input_pos,  # [b, n, 2]
                ):
        x = rearrange(x, 'b c t n-> b t n c')
        x = self.to_embedding(x)
        x = self.dropout(x)

        x, input_pos = self.pooling_layer(x, input_pos)  # [b t n c] -> [b t n//8 c]
        x, x_cls = self.st_transformer.forward(x,
                                             self.cls_token,
                                             self.temp_embedding,
                                             input_pos)
        x = self.shrink_temporal(x)
        x, x_cls = self.expand_feat(x), self.expand_feat(x_cls)
        x, x_cls = self.s_transformer.forward(x, x_cls, input_pos, self.cls_emb)
        x, x_cls = self.project_to_latent(x), self.to_cls(x_cls) * self.gamma

        return x, x_cls, input_pos


class SimpleAttentionEncoder(nn.Module):
    def __init__(self,
                 input_channels,           # how many channels
                 seq_len,                  # this should be the input sequence length
                 in_emb_dim,               # embedding dim of token                 (how about 512)
                 out_seq_emb_dim,          # embedding dim of encoded sequence      (how about 256)
                 depth,                    # depth of transformer / how many layers of attention    (4)
                 n_patch=16,
                 out_grid=64,
                 emb_dropout=0.1,           # dropout of embedding
                 attention_init='xavier',
                 ):
        super().__init__()
        self.n_patch = n_patch
        self.out_grid = out_grid

        t = seq_len
        self.dropout = nn.Dropout(emb_dropout)

        self.to_embedding = nn.Sequential(
            nn.Linear(input_channels, in_emb_dim, bias=False),
            nn.GELU(),
            nn.Linear(in_emb_dim, in_emb_dim, bias=False),
            nn.GELU(),
            nn.Linear(in_emb_dim, in_emb_dim, bias=False),
        )

        self.temp_embedding = nn.Parameter(
            torch.cat((torch.tensor([-1.]), torch.linspace(0, 1, t)), dim=0).view(1, t+1, 1), requires_grad=False)   # [b, t, 1]
        self.cls_token = nn.Parameter(torch.randn(1, 1, in_emb_dim), requires_grad=True)
        self.cls_emb = nn.Parameter(torch.randn(1, 1, 2), requires_grad=True)
        self.gamma = nn.Parameter(torch.tensor([0.1]), requires_grad=True)

        self.st_transformer = STTransformerCat(in_emb_dim, 1, 4, 64, in_emb_dim, 'galerkin', attention_init=attention_init)

        self.s_transformer = TransformerCat(in_emb_dim*2, depth-1, 4, 64, in_emb_dim*2, 'galerkin', attention_init=attention_init)

        self.to_cls = nn.Sequential(
            nn.LayerNorm(2*in_emb_dim),
            nn.Linear(2*in_emb_dim, out_seq_emb_dim, bias=True)
            )

        self.shrink_temporal = nn.Sequential(
            Rearrange('b n t c -> b n (t c)'),
            nn.Linear(t*in_emb_dim, 2*in_emb_dim, bias=False),
        )
        self.expand_feat = nn.Linear(in_emb_dim, 2*in_emb_dim)

        self.project_to_latent = nn.Sequential(
            nn.Linear(2*in_emb_dim, out_seq_emb_dim, bias=False))

    def forward(self,
                x,  # [b, c, t, n]
                input_pos,  # [b, n, 2]
                ):
        x = rearrange(x, 'b c t n-> b t n c')
        x = self.to_embedding(x)
        x = self.dropout(x)

        x, x_cls = self.st_transformer.forward(x,
                                             self.cls_token,
                                             self.temp_embedding,
                                             input_pos)
        x = self.shrink_temporal(x)
        x_cls = self.expand_feat(x_cls)
        x, x_cls = self.s_transformer.forward(x, x_cls, input_pos, self.cls_emb)
        x, x_cls = self.project_to_latent(x), self.to_cls(x_cls) * self.gamma

        return x, x_cls


class NoSTAttentionEncoder(nn.Module):
    def __init__(self,
                 input_channels,           # how many channels
                 seq_len,                  # this should be the input sequence length
                 in_emb_dim,               # embedding dim of token                 (how about 512)
                 out_seq_emb_dim,          # embedding dim of encoded sequence      (how about 256)
                 depth,                    # depth of transformer / how many layers of attention    (4)
                 emb_dropout=0.1,           # dropout of embedding
                 ):
        super().__init__()

        t = seq_len
        self.dropout = nn.Dropout(emb_dropout)

        self.to_embedding = nn.Sequential(
            nn.Linear(input_channels, in_emb_dim, bias=False),
            nn.GELU(),
            nn.Linear(in_emb_dim, in_emb_dim, bias=False),
            nn.GELU(),
            nn.Linear(in_emb_dim, in_emb_dim, bias=False),
        )

        self.cls_token = nn.Parameter(torch.randn(1, 2*in_emb_dim), requires_grad=True)
        self.cls_emb = nn.Parameter(torch.randn(1, 1, 2), requires_grad=True)
        self.gamma = nn.Parameter(torch.tensor([0.1]), requires_grad=True)

        self.shrink_temporal = nn.Sequential(
            Rearrange('b t n c -> b n (t c)'),
            nn.Linear(t * in_emb_dim, 2 * in_emb_dim, bias=False),
        )
        self.s_transformer = TransformerCat(in_emb_dim*2, depth, 4, 64, in_emb_dim*2, 'galerkin', init_scale=32)

        self.to_cls = nn.Sequential(
            nn.Linear(2*in_emb_dim, out_seq_emb_dim, bias=False),
            nn.GELU(),
            nn.Linear(out_seq_emb_dim, out_seq_emb_dim, bias=False),
            nn.GELU(),
            nn.Linear(out_seq_emb_dim, out_seq_emb_dim, bias=False),
            nn.LayerNorm(out_seq_emb_dim))

        self.project_to_latent = nn.Sequential(
            nn.Linear(2*in_emb_dim, out_seq_emb_dim, bias=False),
            nn.GELU(),
            nn.Linear(out_seq_emb_dim, out_seq_emb_dim, bias=False),
            nn.GELU(),
            nn.Linear(out_seq_emb_dim, out_seq_emb_dim, bias=False),
            nn.InstanceNorm1d(in_emb_dim))

    def forward(self,
                x,  # [b, c, t, n]
                input_pos,  # [b, n, 2]
                ):
        x = rearrange(x, 'b c t n-> b t n c')
        x = self.to_embedding(x)
        x = self.dropout(x)
        x = self.shrink_temporal(x)

        x, x_cls = self.s_transformer.forward(x, self.cls_token, input_pos, self.cls_emb)
        x, x_cls = self.project_to_latent(x), self.to_cls(x_cls) * self.gamma

        return x, x_cls


class SpatialTemporalEncoder2D(nn.Module):
    def __init__(self,
                 input_channels,           # how many channels
                 in_emb_dim,               # embedding dim of token                 (how about 512)
                 out_seq_emb_dim,          # embedding dim of encoded sequence      (how about 256)
                 heads,
                 depth,                    # depth of transformer / how many layers of attention    (4)
                 ):
        super().__init__()

        self.to_embedding = nn.Sequential(
            # Rearrange('b c n -> b n c'),
            nn.Linear(input_channels, in_emb_dim, bias=False),
        )

        if depth > 4:
            self.s_transformer = TransformerCatNoCls(in_emb_dim, depth, heads, in_emb_dim, in_emb_dim,
                                                     'galerkin', True, scale=[32, 16, 8, 8] +
                                                                             [1] * (depth - 4),
                                                     attention_init='orthogonal')
        else:
            self.s_transformer = TransformerCatNoCls(in_emb_dim, depth, heads, in_emb_dim, in_emb_dim,
                                                     'galerkin', True, scale=[32] + [16]*(depth-2) + [1],
                                                     attention_init='orthogonal')

        self.project_to_latent = nn.Sequential(
            nn.Linear(in_emb_dim, out_seq_emb_dim, bias=False))

    def forward(self,
                x,  # [b, t(*c)+2, n]
                input_pos,  # [b, n, 2]
                ):

        x = self.to_embedding(x)
        x = self.s_transformer.forward(x, input_pos)
        x = self.project_to_latent(x)

        return x


class SpatialEncoder2D(nn.Module):
    def __init__(self,
                 input_channels,           # how many channels
                 in_emb_dim,               # embedding dim of token                 (how about 512)
                 out_seq_emb_dim,          # embedding dim of encoded sequence      (how about 256)
                 heads,
                 depth,                    # depth of transformer / how many layers of attention    (4)
                 res,
                 use_ln=True,
                 emb_dropout=0.05,           # dropout of embedding
                 ):
        super().__init__()

        self.to_embedding = nn.Sequential(
            nn.Linear(input_channels, in_emb_dim, bias=False),
        )

        self.dropout = nn.Dropout(emb_dropout)

        self.s_transformer = TransformerCatNoCls(in_emb_dim, depth, heads, in_emb_dim, in_emb_dim,
                                                 'galerkin',
                                                 use_relu=False,
                                                 use_ln=use_ln,
                                                 scale=[res, res//4] + [1]*(depth-2),
                                                 relative_emb_dim=2,
                                                 min_freq=1 / res,
                                                 dropout=0.03,
                                                 attention_init='orthogonal')

        self.to_out = nn.Sequential(
            nn.Linear(in_emb_dim, out_seq_emb_dim, bias=False))

    def forward(self,
                x,  # [b, n, c]
                input_pos,  # [b, n, 2]
                ):

        x = self.to_embedding(x)
        x = self.dropout(x)

        x = self.s_transformer.forward(x, input_pos)
        x = self.to_out(x)

        return x


class Encoder1D(nn.Module):
    def __init__(self,
                 input_channels,           # how many channels
                 in_emb_dim,               # embedding dim of token                 (how about 512)
                 out_seq_emb_dim,          # embedding dim of encoded sequence      (how about 256)
                 depth,                    # depth of transformer / how many layers of attention    (4)
                 emb_dropout=0.05,           # dropout of embedding
                 res=2048,
                 ):
        super().__init__()

        # self.dropout = nn.Dropout(emb_dropout)

        self.to_embedding = nn.Sequential(
            nn.Linear(input_channels, in_emb_dim-1, bias=False),
        )

        self.transformer = TransformerCatNoCls(in_emb_dim, depth, 1, in_emb_dim, in_emb_dim, 'fourier',
                                               scale=[8.] + [4.]*2 + [1.]*(depth-3),
                                               relative_emb_dim=1,
                                               min_freq=1/res,
                                               attention_init='orthogonal')

        self.project_to_latent = nn.Sequential(
            nn.Linear(in_emb_dim, out_seq_emb_dim, bias=False))

    def forward(self,
                x,  # [b, n, c]
                input_pos,  # [b, n, 1]
                ):
        x = self.to_embedding(x)
        # x = self.dropout(x)
        x = torch.cat((x, input_pos), dim=-1)
        x = self.transformer.forward(x, input_pos)
        x = self.project_to_latent(x)

        return x


# for ablation
class NoRelSpatialTemporalEncoder2D(nn.Module):
    def __init__(self,
                 input_channels,           # how many channels
                 in_emb_dim,               # embedding dim of token                 (how about 512)
                 out_seq_emb_dim,          # embedding dim of encoded sequence      (how about 256)
                 heads,
                 depth,                    # depth of transformer / how many layers of attention    (4)
                 ):
        super().__init__()

        self.to_embedding = nn.Sequential(
            # Rearrange('b c n -> b n c'),
            nn.Linear(input_channels, in_emb_dim, bias=False),
        )

        if depth > 4:
            self.s_transformer = TransformerCatNoCls(in_emb_dim, depth, heads, in_emb_dim, in_emb_dim,
                                                     'galerkin', True, scale=-1,
                                                     attention_init='orthogonal')
        else:
            self.s_transformer = TransformerCatNoCls(in_emb_dim, depth, heads, in_emb_dim, in_emb_dim,
                                                     'galerkin', True, scale=-1,
                                                     attention_init='orthogonal')

        self.project_to_latent = nn.Sequential(
            nn.Linear(in_emb_dim, out_seq_emb_dim, bias=False))

    def forward(self,
                x,  # [b, t(*c)+2, n]
                input_pos,  # [b, n, 2]
                ):

        x = self.to_embedding(x)
        x = self.s_transformer.forward(x, input_pos)
        x = self.project_to_latent(x)

        return x



