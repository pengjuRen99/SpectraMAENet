import torch
from torch import nn
from torch import einsum
import numpy as np
from einops import rearrange, repeat
from timm.models.layers import DropPath


class CLSToken(nn.Module):
    def __init__(self, dim):
        super(CLSToken, self).__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
    
    def forward(self, x):
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        return x
    

class AbsPosEmbedding(nn.Module):
    def __init__(self, spectra_size, patch_size, dim, stride=None, cls=True):
        super(AbsPosEmbedding, self).__init__()
        if not spectra_size % patch_size == 0:
            raise Exception('Spectra dimensions must be divisible by the patch size!')
        stride = patch_size if stride is None else stride
        output_size = self._conv_output_size(spectra_size, patch_size, stride)
        self.pos_embedding = nn.Parameter(torch.randn(1, output_size + int(cls), dim) * 0.02)
    
    def forward(self, x):
        x = x + self.pos_embedding
        return x
    
    @staticmethod
    def _conv_output_size(spectra_size, kernel_size, stride, padding=0):
        return int(((spectra_size - kernel_size + (2 * padding)) / stride) + 1)
    

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos, cls_token=False):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.         # 将每个位置进行归一化
    omega = 1. / 10000**omega  # (D/2,)

    pos = np.arange(pos, dtype=np.float64)

    # pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)

    if cls_token:
        emb = np.concatenate([np.zeros([1, embed_dim]), emb], axis=0)

    return emb


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super(LayerScale, self).__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))
    
    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class PatchEmbedding(nn.Module):
    def __init__(self, spectra_size, patch_size, embed_dim, norm_layer=None):
        super(PatchEmbedding, self).__init__()
        if not spectra_size % patch_size == 0:
            raise Exception('Spectra dimensions must be divisible by the patch size!')
        self.patch_size = patch_size
        self.num_patches = spectra_size // patch_size
        self.proj = nn.Conv1d(in_channels=1, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        '''
        patch_dim = channel * patch_size
        self.patch_embedding = nn.Sequential(
            Rearrange('b c (d p) -> b d (p c)', p=patch_size),      # b=batchsize, c=channel, d=number of patch, p=size of patch
            nn.Linear(patch_dim, embed_dim),
        )
        '''
    
    def forward(self, x):
        # x = self.patch_embedding(x)
        x = self.proj(x)
        x = x.transpose(1, 2)       # B, N(number of patch), C
        x = self.norm(x)
        return x
    

class Attention1d(nn.Module):
    def __init__(self, dim_in, dim_out=None, heads=8, dim_head=64, dropout=0.0, **kwargs):
        super(Attention1d, self).__init__()
        inner_dim = heads * dim_head
        dim_out = dim_in if dim_out is None else dim_out

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim_in, inner_dim*3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim_out),
            nn.Dropout(dropout) if dropout  > 0.0 else nn.Identity(),
        )

    def forward(self, x, mask=None):
        b, n, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        dots = dots + mask if mask is not None else dots
        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        return out, attn
    

class Mlp(nn.Module):
    def __init__(self, dim_in, hidden_dim=None, dim_out=None, dropout=0.0, f=nn.Linear, activation=nn.GELU, **kwargs):
        super(Mlp, self).__init__()
        dim_out = dim_in if dim_out is None else dim_out
        hidden_features = hidden_dim or dim_in
        self.net = nn.Sequential(
            f(dim_in, hidden_dim),
            activation(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            f(hidden_features, dim_out),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
        )
    
    def forward(self, x):
        x = self.net(x)
        return x
    

class Block(nn.Module):
    def __init__(self, dim_in, dim_out=None, heads=8, dim_mlp=1024, dropout=0.0, sd=0.0, attn=Attention1d, norm=nn.LayerNorm, 
                 f=nn.Linear, activation=nn.GELU, init_values=None, **kwargs):
        super(Block, self).__init__()
        dim_out = dim_in if dim_out is None else dim_out

        self.shortcut = []
        if dim_in != dim_out:
            self.shortcut.append(norm(dim_in))
            self.shortcut.append(nn.Linear(dim_in, dim_out))
        self.shortcut = nn.Sequential(*self.shortcut)

        self.norm1 = norm(dim_in)
        self.attn = attn(dim_in, dim_out, heads=heads, dim_head=int(dim_in//heads), dropout=dropout)
        self.ls1 = LayerScale(dim_out, init_values=init_values) if init_values else nn.Identity()
        self.sd1 = DropPath(sd) if sd > 0.0 else nn.Identity()

        self.norm2 = norm(dim_out)
        self.ff = Mlp(dim_out, dim_mlp, dim_out, dropout=dropout, f=f, activation=activation)
        self.ls2 = LayerScale(dim_out, init_values=init_values) if init_values else nn.Identity()
        self.sd2 = DropPath(sd) if sd > 0.0 else nn.Identity()

    def forward(self, x, mask=None):
        skip = self.shortcut(x)
        x = self.norm1(x)
        x, attn = self.attn(x, mask=mask)
        x = self.ls1(x)
        x = self.sd1(x) + skip

        skip = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.ls2(x)
        x = self.sd2(x) + skip

        return x    


class ViT(nn.Module):
    def __init__(self, spectra_size, patch_size, num_classes, dim, depth, heads, dim_mlp, dropout, 
                 emb_dropout, sd, classifier=None, **kwargs):
        super(ViT, self).__init__()
        self.patch_embed = PatchEmbedding(spectra_size=spectra_size, patch_size=patch_size, embed_dim=dim, )
        # cls_token and position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches+1, dim), requires_grad=False) # 1d sincos pos_embed
        '''
        self.cls_token = CLSToken(dim=dim)      # no specific change, add standard var
        self.pos_embed = AbsPosEmbedding()      # no specific change, add standard var, add cls_token
        '''
        self.dropout = nn.Dropout(emb_dropout) if emb_dropout > 0.0 else nn.Identity()

        self.blocks = []
        for i in range(depth):
            self.blocks.append(
                Block(dim, heads=heads, dim_head=int(dim//heads), dim_mlp=dim_mlp, dropout=dropout, sd=(sd * i / (depth - 1)))
            )
        self.blocks = nn.Sequential(*self.blocks)
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim),
            # nn.Flatten(1,2),
            # nn.Linear(self.patch_embed.num_patches*dim, num_classes),
            # nn.AdaptiveAvgPool1d(dim),
            nn.Linear(dim, num_classes),
        )if classifier is None else classifier
        self.initialize_weights()

    def initialize_weights(self, ):
        # initialize positions embedding
        pos_embed = get_1d_sincos_pos_embed_from_grid(self.pos_embed.shape[-1], self.patch_embed.num_patches, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        # initialize patch_embed like nn.Linear (instead of Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=0.02)
        # initialize nn.Linear and nn.Layernorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = self.blocks(x)
        # x = self.classifier(x)
        x = self.classifier(x[:, 0])

        return x
    

def spectra_ViT1D_patch100_dim128_h4(**kwargs):
    model = ViT(
        spectra_size=1000, patch_size=100, num_classes=30, dim=128, depth=4, heads=4, dim_mlp=256, 
        dropout=0.1, emb_dropout=0.1, sd=0.1, 
    )
    return model

spectra_ViT1D_patch100 = spectra_ViT1D_patch100_dim128_h4
    