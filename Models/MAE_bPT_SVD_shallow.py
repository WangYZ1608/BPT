import torch, copy
import torch.nn as nn
from .MAE_vit import VisionTransformer
from functools import partial
from torchsummaryX import summary
import pdb

__all__ = ['MAE_bpt_svd_vit_b', 'MAE_bpt_svd_vit_l', 'MAE_bpt_svd_vit_h']


class PromptVisionTransformer(VisionTransformer):
    def __init__(self, num_prompts=100, t=2.0, channels=768, global_pool=True, **kwargs):
        super(PromptVisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool

        embed_dim = kwargs['embed_dim']
        self.t = t
        self.random_vectors = nn.Parameter(torch.zeros(num_prompts, embed_dim), requires_grad=True)
            
        torch.nn.init.normal_(self.random_vectors, std=.02)
    
    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        u, s1, v = torch.svd(self.random_vectors)
        s2 = torch.sign(s1) * (s1 ** self.t)
        p_tokens = u @ torch.diag(s2) @ v.transpose(1, 0)

        prompt_tokens = p_tokens.unsqueeze(dim=0).expand(B, -1, -1)
        x = torch.cat((
            x[:, :1, :],
            prompt_tokens,
            x[:, 1:, :]
            ), dim=1)
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            out = self.norm(x)
        else:
            x = self.norm(x)
            out = x[:, 0]

        return out, p_tokens
    
    def forward_head(self, x):
        x = self.head(x)
        return x

    def forward(self, x):
        x, p_token = self.forward_features(x)
        logits = self.forward_head(x)
        
        return logits, p_token


def MAE_bpt_svd_vit_b(**kwargs):
    model = PromptVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def MAE_bpt_svd_vit_l(**kwargs):
    model = PromptVisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def MAE_bpt_svd_vit_h(**kwargs):
    model = PromptVisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model