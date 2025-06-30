import torch
import torch.nn as nn
import timm.models.vision_transformer
from functools import partial
from torchsummaryX import summary
import pdb

__all__ = ['vit_base_patch16', 'vit_large_patch16', 'vit_huge_patch14']

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    def __init__(self, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)
    
    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for ids, blk in enumerate(self.blocks):
            if ids == 11:
                x, x_ = blk(x)
            else:
                x, _ = blk(x)

        x = self.norm(x)
        out = x[:, 0]

        return out, x_
    
    def forward_head(self, x):
        x = self.head(x)
        return x

    def forward(self, x):
        x, x_ = self.forward_features(x)
        logits = self.forward_head(x)
        return logits, x_


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# model = vit_base_patch16()
# summary(model, torch.ones(2, 3, 224, 224))