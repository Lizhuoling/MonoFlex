import math
import pdb
from functools import partial

import torch
import torch.nn as nn

def build_vit(cfg, **kwargs):
    img_size = (cfg.INPUT.HEIGHT_TRAIN, cfg.INPUT.WIDTH_TRAIN)
    model = VisionTransformer(
        img_size = img_size, patch_size=cfg.MODEL.BACKBONE.DOWN_RATIO, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)     # atten shape: (B, num_heads, N, N)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(self, dim, num_heads, input_resolution, window_size = 4, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        '''
        Input:
            x shape: (B, L, C) = (B, Hp * Wp) = (B, H/patch_size * W/patch_size, C)
        '''
        B, L, C = x.shape
        assert L == self.input_resolution[0] * self.input_resolution[1]
        windows_x = x.reshape(B, self.input_resolution[0], self.input_resolution[1], C)
        windows_x = self.window_partition(windows_x, window_size = self.window_size)    # Left x shape: (B * num_windows, window_size, window_size, C)
        windows_x = windows_x.view(-1, self.window_size * self.window_size, C)
        y, attn = self.attn(self.norm1(windows_x))  # y shape: (B * num_windows, window_size * window_size, C). attn shape: (B * num_windows, window_size * window_size, window_size * window_size)
        y = self.window_restore(y, self.input_resolution, self.window_size) # Left y shape: (B, Hp * Wp, C)
        if return_attention:
            attn = self.window_restore(attn, self.input_resolution, self.window_size)   # Left attn shape: (B, Hp * Wp, window_size * window_size)
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    
    def window_partition(self, x, window_size):
        """
        Description:
            Devide the input tensor with shape: (B, H, W, C) as (B * H/window_size * W/window_size, window_size, window_size, C)
        Input
            x: (B, H, W, C)
            window_size (int): window size
        Output:
            windows: (B * num_windows, window_size, window_size, C)
        """
        B, H, W, C = x.shape
        assert H % window_size ==0 and W % window_size == 0
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
        return windows

    def window_restore(self, x, ori_resolution, window_size):
        '''
        Description:
            Restore the window partitioned x back.
        Input:
            x: The embbeding produced by attention block. shape: (B * num_windows, window_size * window_size, C).
        Output:
            y shape: (B, Hp * Wp, C) = (B, H/patch_size * W/patch_size, C)
        '''
        assert len(ori_resolution) == 2 and len(x.shape) == 3
        C = x.shape[2]
        H_window_num, W_window_num = ori_resolution[0] // window_size, ori_resolution[1] // window_size
        x = x.reshape(-1, H_window_num, W_window_num, window_size, window_size, C)
        B = x.shape[0]
        x = x.permute(0, 1, 3, 2, 4, 5) # Left x shape: (B, H // window_size, window_size, W // window_size, window_size, C)
        y = x.reshape(B, ori_resolution[0] * ori_resolution[1], C)
        return y


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=(224, 224), patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        if (img_size[0] % patch_size != 0) or (img_size[1] % patch_size != 0):
            raise Exception('The size of image must be divisible by patch_size.')
        num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x    # x shape: (B, L, C) = (B, H/patch_size * W/patch_size, C)


class VisionTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(self, img_size, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.patch_size = patch_size
        self.num_features = self.embed_dim = embed_dim
        self.out_channels = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        
        block_input_resolution = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, input_resolution = block_input_resolution, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        
        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)  
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding. left x shape: (B, patch_num, C) = (B, H/patch_size * W/patch_size, C)
        # add positional encoding to each token
        x = x + self.pos_embed
        return self.pos_drop(x)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.prepare_tokens(x)  # Right x shape: (B, C, H, W). Left x shape: (B, patch_num, C)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)    # Left x shape: (B, patch_num, C)
        x = x.reshape(B, H // self.patch_size, W // self.patch_size, -1)
        return x.permute(0, 3, 1, 2)   # Convert the tensor format as (B, C, H, W)

    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)

    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output