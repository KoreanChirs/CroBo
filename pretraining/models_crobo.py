from functools import partial
import random

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block
from timm.models.vision_transformer import CrossAttention, Attention, DropPath, Mlp

from util.pos_embed import get_2d_sincos_pos_embed, get_sinusoid_encoding_table


class CrossAttention_crobo(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        num_frames=2,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, kvx, num_frames=1, src_mask=None):
        B, N, C = x.shape
        _, kvN, _ = kvx.shape
        if src_mask != None:
            kv = (
                self.kv(kvx)
                .reshape(B // num_frames, kvN, 2, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
        else:
            kv = (
                self.kv(kvx)
                .reshape(B, kvN, 2, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
        q = (
            self.q(x)
            .reshape(B, N, 1, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )

        q, k, v = q[0], kv[0], kv[1]

        if src_mask != None:
            k = k.repeat(num_frames, 1, 1, 1)
            v = v.repeat(num_frames, 1, 1, 1)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if src_mask != None:
            src_mask = src_mask.unsqueeze(1).unsqueeze(1)
            src_mask = src_mask.repeat(1, self.num_heads, N, 1)
            attn = attn.masked_fill(src_mask == 0, -1e4)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CSABlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        num_frames=2,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm_kv1 = norm_layer(dim)
        self.cattn = CrossAttention_crobo(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            num_frames=num_frames,
        )
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm3 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.norm4 = norm_layer(dim)
        self.mlp2 = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, kvx, num_frames=1, src_mask=None, path="sm"):
        if path == "csm":
            x = x + self.drop_path(
                self.cattn(self.norm1(x), self.norm_kv1(kvx), src_mask=src_mask, num_frames=num_frames)
            )
            x = x + self.drop_path(self.attn(self.norm3(x)))
        elif path == "sm":
            x = x + self.drop_path(self.attn(self.norm3(x)))
        elif path == "scm":
            x = x + self.drop_path(self.attn(self.norm3(x)))
            x = x + self.drop_path(
                self.cattn(self.norm1(x), self.norm_kv1(kvx), src_mask=src_mask, num_frames=num_frames)
            )
        x = x + self.mlp2(self.norm4(x))
        return x


class crobo(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=True,
        mask_ratio=0.9,
        mask_ratio_src=0.,
        batch_size=None,
        repeated_sampling=2,
        crobo_path=None,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    qk_scale=None,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False
        )

        self.decoder_blocks = nn.ModuleList(
            [
                CSABlock(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    qk_scale=None,
                    norm_layer=norm_layer,
                )
                for i in range(decoder_depth)
            ]
        )

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size**2 * in_chans, bias=True
        )

        self.norm_pix_loss = norm_pix_loss
        self.mask_ratio = mask_ratio
        self.mask_ratio_src = mask_ratio_src
        self.batch_size = batch_size * repeated_sampling
        self.crobo_path = crobo_path

        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=True,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=True,
        )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
        )

        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_keep

    def forward_encoder(self, imgs, mask_ratio=0.0):
        x = self.patch_embed(imgs)
        x = x + self.pos_embed[:, 1:, :]

        if mask_ratio != 0.0:
            x, mask, ids_restore, ids_keep = self.random_masking(x, mask_ratio)
        else:
            mask, ids_restore, ids_keep = None, None, None

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x, mask, ids_restore, ids_keep

    def forward_decoder_crobo(self, src_h, tgt_h, ids_restore):
        # Use only CLS token from global view as context
        src_cls = self.decoder_embed(src_h[:, :1])  # [B, 1, decoder_dim]
        src_cls = src_cls + self.decoder_pos_embed[:, :1]

        # Project local view tokens
        tgt_h = self.decoder_embed(tgt_h)

        # Append mask tokens and unshuffle
        mask_tokens = self.mask_token.repeat(tgt_h.shape[0], ids_restore.shape[1] + 1 - tgt_h.shape[1], 1)
        tgt_ = torch.cat([tgt_h[:, 1:, :], mask_tokens], dim=1)  # no cls token
        tgt_ = torch.gather(tgt_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, tgt_h.shape[2]))  # unshuffle
        tgt_ = tgt_ + self.decoder_pos_embed[:, 1:]

        # Replace CLS with global CLS (bottleneck token)
        x = torch.cat([src_cls, tgt_], dim=1)

        for blk in self.decoder_blocks:
            x = blk(x, kvx=None, path=self.crobo_path)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        x = x[:, 1:, :]  # remove cls token

        return x

    def forward_loss(self, imgs, pred, mask=None):
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        recon_loss = (pred - target) ** 2
        if mask is not None:
            recon_loss = recon_loss.mean(dim=-1)
            recon_loss = (recon_loss * mask).sum() / mask.sum()
        else:
            recon_loss = recon_loss.mean()

        return recon_loss

    def forward(self, list_imgs, epoch, list_idx=None):
        src_imgs = list_imgs[0]   # global view
        tgt_imgs = list_imgs[-1]  # local view

        src_h, _, _, _ = self.forward_encoder(src_imgs, mask_ratio=0.0)
        tgt_h, tgt_mask, tgt_ids_restore, _ = self.forward_encoder(tgt_imgs, mask_ratio=self.mask_ratio)

        pred = self.forward_decoder_crobo(src_h, tgt_h, tgt_ids_restore)
        loss = self.forward_loss(tgt_imgs, pred, tgt_mask)

        return loss


def crobo_vit_small_patch16_dec512d8b(**kwargs):
    model = crobo(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model

def crobo_vit_base_patch16_dec512d8b(**kwargs):
    model = crobo(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model

def crobo_vit_large_patch16_dec512d8b(**kwargs):
    model = crobo(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model

crobo_vit_small_patch16 = crobo_vit_small_patch16_dec512d8b
crobo_vit_base_patch16 = crobo_vit_base_patch16_dec512d8b
crobo_vit_large_patch16 = crobo_vit_large_patch16_dec512d8b
