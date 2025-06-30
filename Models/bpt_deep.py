import torch
import torch.nn as nn
import numpy as np
from .vit import VisionTransformer
from functools import partial
from torchsummaryX import summary
import pdb

__all__ = ['bpt_deep_vit_b', 'bpt_deep_vit_l', 'bpt_deep_vit_h']


class PromptVisionTransformer(VisionTransformer):
    def __init__(self, num_prompts=100, channels=100, **kwargs):
        super(PromptVisionTransformer, self).__init__(**kwargs)

        self.num_prompts = num_prompts

        self.embed_dim = kwargs['embed_dim']
        self.depth = kwargs['depth']
        self.channels = channels
        self.random_vectors = nn.Parameter(torch.zeros(self.depth, num_prompts, channels), requires_grad=True)
        
        # conv module
        self.conv1x1 = nn.Sequential(*[nn.Conv2d(in_channels=channels, out_channels=self.embed_dim, kernel_size=1, bias=False) for _ in range(self.depth)])

        # init
        torch.nn.init.normal_(self.random_vectors, std=.02)
        self._init_weights(self.conv1x1)

    def _init_weights(self, m):
        q_path = ["./MoCo_X/qk_weights/MoCo_" + str(i+1) + "_q_weights.npy" for i in range(self.depth)]
        k_path = ["./MoCo_X/qk_weights/MoCo_" + str(i+1) + "_k_weights.npy" for i in range(self.depth)]
        x_path = ["./MoCo_X/x_cubs/x" + str(i) + ".npy" for i in range(self.depth)]

        for ids, conv in enumerate(m):
            assert isinstance(conv, nn.Conv2d)
            q = np.load(q_path[ids])
            k = np.load(k_path[ids])
            x = np.load(x_path[ids])

            q = torch.as_tensor(q, dtype=torch.float32)
            k = torch.as_tensor(k, dtype=torch.float32)
            x = torch.as_tensor(x, dtype=torch.float32)

            qkx = q @ k.transpose(1, 0) @ x.transpose(2, 1)
            qkx = qkx.mean(dim=0)
            
            n = qkx.size(0)
            sigma = qkx @ qkx.transpose(1, 0) * (1 /n)
            u, s, v = torch.svd(sigma)

            w = u @ torch.diag(s ** (-0.5)) @ v.transpose(1, 0)

            noise = torch.rand(self.embed_dim, self.embed_dim)   # noise in [0, 1]
            ids_shuffle = torch.argsort(noise, dim=1)
            ids_keep = ids_shuffle[:, :self.channels]
            w_ = torch.gather(w, dim=1, index=ids_keep)
            assert w_.size() == (self.embed_dim, self.channels)

            conv.weight.data = w_.reshape(w_.size(0), w_.size(1), 1, 1).clone()

    def _creat_prompts(self, x: torch.Tensor):
        B, length, dim = x.size()
        prompt_length = int(np.sqrt(self.num_prompts))

        prompt_tokens = []
        for ids in range(self.depth):
            vectors = self.random_vectors[ids].unsqueeze(0).expand(B, -1, -1).transpose(1, 2).contiguous().reshape(B, -1, prompt_length, prompt_length)
            vectors = self.conv1x1[ids](vectors)
            prompt_tokens.append(vectors.reshape(B, dim, -1).transpose(1, 2).contiguous())
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
            prompt_tokens[0],
            x[:, 1:, :]
            ), dim=1)
        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            if i == 0:
                x = blk(x)
            else:
                x = torch.cat((
                    x[:, :1, :],
                    prompt_tokens[i],
                    x[:, 1+self.num_prompts:, :]
                    ), dim=1)
                x = blk(x)

        x = self.norm(x)
        out = x[:, 0]

        return out


def bpt_deep_vit_b(**kwargs):
    model = PromptVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def bpt_deep_vit_l(**kwargs):
    model = PromptVisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def bpt_deep_vit_h(**kwargs):
    model = PromptVisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
