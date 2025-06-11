import torch.nn as nn
from model.ViT_1D import PatchEmbedding, Block
import torch
from model.ViT_1D import get_1d_sincos_pos_embed_from_grid


class MaskedAutoencoderViT(nn.Module):
    '''
        Masked Autoencoder with VisionTransformer backone
    '''
    def __init__(self, spectra_size=1000, patch_size=200, encoder_dim=128, depth=4, num_heads=8, 
                 decoder_embed_dim=256, decoder_depth=6, decoder_num_heads=8, dim_mlp=256, norm_layer=nn.LayerNorm, 
                 norm_pix_loss=False):
        super(MaskedAutoencoderViT, self).__init__()
        self.patch_size = patch_size
        # ----------------------------------------------------------------
        # Spectra_MAE encoder specifics
        self.patch_embed = PatchEmbedding(spectra_size=spectra_size, patch_size=patch_size, embed_dim=encoder_dim, )
        self.num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, encoder_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches+1, encoder_dim), requires_grad=False)
        self.blocks = nn.ModuleList([
            Block(dim_in=encoder_dim, heads=num_heads, dim_head=int(encoder_dim//num_heads), dim_mlp=dim_mlp)
            for i in range(depth)
        ])
        self.norm = norm_layer(encoder_dim)

        # ----------------------------------------------------------------
        # Spectra_MAE decoder specifics
        self.decoder_embed = nn.Linear(encoder_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches+1, decoder_embed_dim), requires_grad=False)       # fixed sin-cos embedding
        self.decoder_blocks = nn.ModuleList([
            Block(dim_in=decoder_embed_dim, heads=decoder_num_heads, dim_head=int(decoder_embed_dim//decoder_num_heads), 
                  dim_mlp=dim_mlp)
                  for i in range(decoder_depth)
        ])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size, bias=True)

        # ----------------------------------------------------------------
        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()

    def initialize_weights(self):
        # initialize positions embedding 
        pos_embed = get_1d_sincos_pos_embed_from_grid(self.pos_embed.shape[-1], self.num_patches, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        decoder_pos_embed = get_1d_sincos_pos_embed_from_grid(self.decoder_pos_embed.shape[-1], self.num_patches, cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        # initialize patch_embed like nn.Linear (instead of Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)
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

    
    def patchify(self, spectra):
        '''
            spectra: [B, 1, L]
            x: [B, L/patch_size, patch_size]
        '''
        assert spectra.shape[2] % self.patch_size == 0

        n = spectra.shape[2] // self.patch_size
        x = spectra.reshape(shape = (spectra.shape[0], 1, n, self.patch_size))
        x = torch.einsum('bcnp->bnpc', x)
        x = x.reshape(shape=(spectra.shape[0], n, self.patch_size))

        return x
    
    def unpatchify(self, x):
        '''
            x: [B, L/patch_size, patch_size]
            spectra: [B, 1, L]
        '''
        p = self.patch_size
        n = self.num_patches
        x = x.reshape(shape=[x.shape[0], n ,p, 1])
        x = torch.einsum('bnpc->bcnp', x)
        spectra = x.reshape(shape=(x.shape[0], 1, n*p))

        return spectra
    
    def random_masking(self, x, mask_ratio):
        '''
            Perform per-sample random masking by per-sample shuffling
            Per-sample shuffling is done by argsort random noise
            x: [N, L, D], sequence
        '''
        N = x.shape[0]
        L = self.num_patches
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(N, L, device=x.device)

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)   # record noise that from small to large
        ids_restore = torch.argsort(ids_shuffle, dim=1)     # order noise

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, x.shape[-1]))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
    
    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))   # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)         # append cls token
        # add pos embed
        x = x + self.decoder_pos_embed
        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        # predictor projection
        x = self.decoder_pred(x)
        # remove cls token
        x = x[:, 1:, :]

        return x
    
    def forward_loss(self, spectra, pred, mask):
        '''
            spectra: [N, 1, L]
            pred: [N, L/patch_size, patch_size]
            mask: [N, L], 0 is keep, 1 is remove
        '''
        target = self.patchify(spectra)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)        # [B, L] mean loss per patch
        loss = (loss * mask).sum() / mask.sum()     # mask not compute loss 

        # loss = loss.sum() / (loss.shape[-1] * loss.shape[-2])         # overall situation compute loss

        return loss
    
    def forward(self, spectra, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(spectra, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(spectra, pred, mask)

        return loss, pred, mask
    

def spectraMAE_patch100_dim256_h8(**kwargs):
    model = MaskedAutoencoderViT(
        spectra_size=1000, patch_size=100, encoder_dim=256, depth=8, num_heads=8, decoder_embed_dim=256, 
        decoder_depth=1, decoder_num_heads=8, dim_mlp=256, 
    )

    return model


spectraMAE_base_patch100 = spectraMAE_patch100_dim256_h8
