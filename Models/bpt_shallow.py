import torch
import torch.nn as nn
import numpy as np
from .vit import VisionTransformer
from functools import partial
from torchsummaryX import summary
import pdb

__all__ = ['bpt_vit_b', 'bpt_vit_l', 'bpt_vit_h']


class PromptVisionTransformer(VisionTransformer):
    def __init__(self, num_prompts=100, channels=100, whitening=False, **kwargs):
        super(PromptVisionTransformer, self).__init__(**kwargs)

        self.embed_dim = kwargs['embed_dim']
        self.num_prompts = num_prompts
        self.channels = channels
        self.random_vectors = nn.Parameter(torch.zeros(1, num_prompts, channels), requires_grad=True)

        # conv module
        self.conv1x1 = nn.Conv2d(in_channels=channels, out_channels=self.embed_dim, kernel_size=1, bias=False)

        # init
        torch.nn.init.normal_(self.random_vectors, std=.02)
        if whitening:
            self._init_weights(self.conv1x1)
        else:
            # random init.
            w = self.conv1x1.weight.data
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
    
    def _init_weights(self, m):
        assert isinstance(m, nn.Conv2d)
        q = np.load("./MoCo_X/qk_weights/MoCo_1_q_weights.npy")
        k = np.load("./MoCo_X/qk_weights/MoCo_1_k_weights.npy")
        x = np.load("./MoCo_X/x_cubs/x0.npy")
        
        q = torch.as_tensor(q, dtype=torch.float32)
        k = torch.as_tensor(k, dtype=torch.float32)
        x = torch.as_tensor(x, dtype=torch.float32)

        qkx = q @ k.transpose(1, 0) @ x.transpose(2, 1)
        qkx = qkx.mean(dim=0)

        n = qkx.size(0)
        sigma = qkx @ qkx.transpose(1, 0) * (1 / n)
        u, s, v = torch.svd(sigma)

        w = u @ torch.diag(s ** (-0.5)) @ v.transpose(1, 0)

        noise = torch.rand(self.embed_dim, self.embed_dim)   # noise in [0, 1]
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_keep = ids_shuffle[:, :self.channels]
        w_ = torch.gather(w, dim=1, index=ids_keep)
        
        m.weight.data = w_.reshape(w_.size(0), w_.size(1), 1, 1).clone()
    
    def _creat_prompts(self, x: torch.Tensor):
        B, length, dim = x.size()
        prompt_length = int(np.sqrt(self.num_prompts))

        vectors = self.random_vectors.expand(B, -1, -1).transpose(1, 2).contiguous().reshape(B, -1, prompt_length, prompt_length)
        vectors = self.conv1x1(vectors)
        prompt_tokens = vectors.reshape(B, dim, -1).transpose(1, 2).contiguous()
        return prompt_tokens
       
    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        
        prompt_tokens = self._creat_prompts(x)

        x = torch.cat((
            x[:, :1, :],
            prompt_tokens,
            x[:, 1:, :]
            ), dim=1)
        x = self.pos_drop(x)
        
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        out = x[:, 0]

        return out


def bpt_vit_b(**kwargs):
    model = PromptVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def bpt_vit_l(**kwargs):
    model = PromptVisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def bpt_vit_h(**kwargs):
    model = PromptVisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model