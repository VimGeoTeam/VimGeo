# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
import torchvision
from functools import partial
from torch import Tensor
from typing import Optional

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, lecun_normal_

from timm.models.layers import DropPath, to_2tuple
from timm.models.vision_transformer import _load_weights

import math

from collections import namedtuple

from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.utils.generation import GenerationMixin
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf
from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
from rope import *
import random
import numpy as np
import matplotlib.pyplot as plt
import os
import csv

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


__all__ = [
    'vim_tiny_patch16_224', 'vim_small_patch16_224', 'vim_base_patch16_224',
    'vim_tiny_patch16_384', 'vim_small_patch16_384', 'vim_base_patch16_384',
]


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding with support for non-square image sizes """
    def __init__(self, img_size=(224, 224), patch_size=(16, 16), stride=(16, 16), in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        # Ensure img_size, patch_size, and stride are tuples
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)

        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = ((img_size[0] - patch_size[0]) // stride[0] + 1, (img_size[1] - patch_size[1]) // stride[1] + 1)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

    

class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False,drop_path=0.,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        # import ipdb; ipdb.set_trace()
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            if residual is None:
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            else:
                hidden_states, residual = fused_add_norm_fn(
                    self.drop_path(hidden_states),
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )    
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


def create_block(
    d_model,
    d_state=16,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    drop_path=0.,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
    if_bimamba=False,
    bimamba_type="none",
    if_divide_out=False,
    init_layer_scale=None,
):
    if if_bimamba:
        bimamba_type = "v1"
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    # import ipdb; ipdb.set_trace()
    mixer_cls = partial(Mamba, d_state=d_state, layer_idx=layer_idx, bimamba_type=bimamba_type, if_divide_out=if_divide_out, init_layer_scale=init_layer_scale, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


class VisionMamba(nn.Module):
    def __init__(self, 
                 mapping_file="./map/mapping.csv",  # 将映射文件作为参数传入
                 save=None,
                 crop=False,
                 img_size=(224, 224),  # 修改为元组
                 patch_size=(16, 16),  # 确保 patch_size 也是元组
                 stride=(16, 16),  # 修改 stride 以支持长宽不同
                 depth=24, 
                 embed_dim=192, 
                 d_state=16,
                 channels=3, 
                 num_classes=1000,
                 ssm_cfg=None, 
                 drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_epsilon: float = 1e-5, 
                 rms_norm: bool = True, 
                 initializer_cfg=None,
                 fused_add_norm=True,
                 residual_in_fp32=True,
                 device=None,
                 dtype=None,
                 ft_seq_len=None,
                 pt_hw_seq_len=14,
                 if_bidirectional=False,
                 final_pool_type='none',
                 if_abs_pos_embed=True,
                 if_rope=False,
                 if_rope_residual=False,
                 flip_img_sequences_ratio=-1.,
                 if_bimamba=False,
                 bimamba_type="v2",
                 if_cls_token=True,
                 if_divide_out=True,
                 init_layer_scale=None,
                 use_double_cls_token=False,
                 use_middle_cls_token=True,
                 **kwargs):
        factory_kwargs = {"device": device, "dtype": dtype}
        # add factory_kwargs into kwargs
        kwargs.update(factory_kwargs) 
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.if_bidirectional = if_bidirectional
        self.final_pool_type = final_pool_type
        self.if_abs_pos_embed = if_abs_pos_embed
        self.if_rope = if_rope
        self.if_rope_residual = if_rope_residual
        self.flip_img_sequences_ratio = flip_img_sequences_ratio
        self.if_cls_token = if_cls_token
        self.use_double_cls_token = use_double_cls_token
        self.use_middle_cls_token = use_middle_cls_token
        self.num_tokens = 1 if if_cls_token else 0
        self.mapping = self.load_mapping_from_csv(mapping_file)  # 加载映射一次并存储为类成员
        # pretrain parameters
        self.num_classes = num_classes
        self.d_model = self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        # 确保传入的是元组
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, stride=stride, in_chans=channels, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        if if_cls_token:
            if use_double_cls_token:
                self.cls_token_head = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
                self.cls_token_tail = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
                self.num_tokens = 2
            else:
                self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
                # self.num_tokens = 1
            
        if if_abs_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, self.embed_dim))
            self.pos_drop = nn.Dropout(p=drop_rate)

        if if_rope:
            half_head_dim = embed_dim // 2
            hw_seq_len = img_size // patch_size
            self.rope = VisionRotaryEmbeddingFast(
                dim=half_head_dim,
                pt_seq_len=pt_hw_seq_len,
                ft_seq_len=hw_seq_len
            )
        # self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.save = save
        self.crop = crop
        # TODO: release this comment
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # import ipdb;ipdb.set_trace()
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
                # transformer blocks
        self.layers = nn.ModuleList(
            [
                create_block(
                    embed_dim,
                    d_state=d_state,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    if_bimamba=if_bimamba,
                    bimamba_type=bimamba_type,
                    drop_path=inter_dpr[i],
                    if_divide_out=if_divide_out,
                    init_layer_scale=init_layer_scale,
                    **factory_kwargs,
                )
                for i in range(depth)
            ]
        )
        
        # output head
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            embed_dim, eps=norm_epsilon, **factory_kwargs
        )
        # self.norm_f_final = (nn.LayerNorm if not rms_norm else RMSNorm)(
        #     12, eps=norm_epsilon, **factory_kwargs
        # )

        # self.pre_logits = nn.Identity()

        # original init
        self.patch_embed.apply(segm_init_weights)
        # self.head.apply(segm_init_weights)
        if if_abs_pos_embed:
            trunc_normal_(self.pos_embed, std=.02)
        if if_cls_token:
            if use_double_cls_token:
                trunc_normal_(self.cls_token_head, std=.02)
                trunc_normal_(self.cls_token_tail, std=.02)
            else:
                trunc_normal_(self.cls_token, std=.02)

        # mamba init
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
    def load_mapping_from_csv(self, filename="mapping.csv"):
        mapping = []
        with open(filename, mode='r', newline='') as file:
            reader = csv.reader(file)
            next(reader)  # 跳过表头
            for row in reader:
                orig_pos = eval(row[0])  # 将字符串形式的元组转换为元组
                new_pos = eval(row[1])
                mapping.append((orig_pos, new_pos))
        return mapping
    def apply_mapping(self, hidden_states):
        """
        Rearrange the input tensor based on quadrant logic with rotations for specific quadrants.
        :param hidden_states: [B, N, C], input feature map
        :return: [B, N, C], rearranged feature map
        """
        B, N, C = hidden_states.shape  # [batch_size, num_patches, feature_dim]

        # Reshape 256 patches into 16x16 grid
        size = int(np.sqrt(N))
        reshaped_states = hidden_states.view(B, size, size, C)  # [B, 16, 16, C]

        # Split into 4 quadrants
        bottom_left = reshaped_states[:, size // 2:, :size // 2, :]
        top_left = reshaped_states[:, :size // 2, :size // 2, :]
        top_right = reshaped_states[:, :size // 2, size // 2:, :]
        bottom_right = reshaped_states[:, size // 2:, size // 2:, :]

        # Rotate leftmost blocks (bottom-left) clockwise
        bottom_left_rotated = bottom_left.permute(0, 2, 1, 3).flip(2)  # Clockwise rotation

        # Rotate rightmost blocks (bottom-right) counterclockwise
        bottom_right_rotated = bottom_right.permute(0, 2, 1, 3).flip(1)  # Counterclockwise rotation

        # Top-left and top-right blocks remain unchanged
        top_left_unchanged = top_left
        top_right_unchanged = top_right

        # Concatenate quadrants into a 1:4 aspect ratio
        rearranged_tensor = torch.cat((
            bottom_left_rotated.squeeze(-1),  # Bottom-left block (rotated clockwise)
            top_left_unchanged.squeeze(-1),  # Top-left block (unchanged)
            top_right_unchanged.squeeze(-1),  # Top-right block (unchanged)
            bottom_right_rotated.squeeze(-1)  # Bottom-right block (rotated counterclockwise)
        ), dim=2)  # Concatenate along width axis

        # Flatten back to [B, N, C]
        final_states = rearranged_tensor.view(B, N, C)

        return final_states

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token", "cls_token_head", "cls_token_tail"}

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)

    def forward_features(self, x, atten=None, indexes=None, inference_params=None, if_random_cls_token_position=False, if_random_token_rank=False):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        _, _, H, W = x.shape# with slight modifications to add the dist_token
        x = self.patch_embed(x)
        B, M, _ = x.shape

        if self.if_cls_token:
            if self.use_double_cls_token:
                cls_token_head = self.cls_token_head.expand(B, -1, -1)
                cls_token_tail = self.cls_token_tail.expand(B, -1, -1)
                token_position = [0, M + 1]
                x = torch.cat((cls_token_head, x, cls_token_tail), dim=1)
                M = x.shape[1]
            else:
                if self.use_middle_cls_token:
                    cls_token = self.cls_token.expand(B, -1, -1)
                    token_position = M // 2
                    # add cls token in the middle
                    x = torch.cat((x[:, :token_position, :], cls_token, x[:, token_position:, :]), dim=1)
                elif if_random_cls_token_position:
                    cls_token = self.cls_token.expand(B, -1, -1)
                    token_position = random.randint(0, M)
                    x = torch.cat((x[:, :token_position, :], cls_token, x[:, token_position:, :]), dim=1)
                    print("token_position: ", token_position)
                else:
                    cls_token = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
                    token_position = 0
                    x = torch.cat((cls_token, x), dim=1)
                M = x.shape[1]

        if self.if_abs_pos_embed and atten is None:
            x = x + self.pos_embed
            x = self.pos_drop(x)
        # 处理裁剪逻辑
        if self.crop and atten is not None:
            grid_size = (x.shape[-2] // self.patch_embed.patch_size[0], x.shape[-1] // self.patch_embed.patch_size[1])
            atten_reshape = torch.nn.functional.interpolate(atten.detach(), grid_size, mode='bilinear')
            order = torch.argsort(atten_reshape[:, 0, :, :].reshape([B, -1]), dim=1)
            select_list = []
            pos_list = []
            crop_rate = 0.64  # 设定裁剪率，可以根据需要调整
            for k in range(B):
                select_list.append(x[k, order[[k], -int(crop_rate * order.shape[1]):]])
                pos_list.append(torch.cat([self.pos_embed[:, :2], self.pos_embed[:, 2 + order[k, -int(crop_rate * order.shape[1]):]]], dim=1))
            x = torch.cat(select_list, dim=0)
            pos_embed = torch.cat(pos_list, dim=0)
            x = x + pos_embed  # 重新加入位置嵌入
        # Handle cropping based on attention map if 'atten' is provided
        if atten is not None:
            grid_size = (x.shape[-2] // self.patch_embed.patch_size[0], x.shape[-1] // self.patch_embed.patch_size[1])
            atten_reshape = torch.nn.functional.interpolate(atten.detach(), grid_size, mode='bilinear')
            order = torch.argsort(atten_reshape[:, 0, :, :].reshape([B, -1]), dim=1)
            select_list = []
            pos_list = []
            crop_rate = 0.64  # Example crop rate; adjust as needed
            for k in range(B):
                select_list.append(x[k, order[[k], -int(crop_rate * order.shape[1]):]])
                pos_list.append(torch.cat([self.pos_embed[:, :2], self.pos_embed[:, 2 + order[k, -int(crop_rate * order.shape[1]):]]], dim=1))
            x = torch.cat(select_list, dim=0)
            pos_embed = torch.cat(pos_list, dim=0)
            x = x + pos_embed  # Optionally add positional embedding here
            
        
        if if_random_token_rank:

            # 生成随机 shuffle 索引
            shuffle_indices = torch.randperm(M)

            if isinstance(token_position, list):
                print("original value: ", x[0, token_position[0], 0], x[0, token_position[1], 0])
            else:
                print("original value: ", x[0, token_position, 0])
            print("original token_position: ", token_position)

            # 执行 shuffle
            x = x[:, shuffle_indices, :]

            if isinstance(token_position, list):
                # 找到 cls token 在 shuffle 之后的新位置
                new_token_position = [torch.where(shuffle_indices == token_position[i])[0].item() for i in range(len(token_position))]
                token_position = new_token_position
            else:
                # 找到 cls token 在 shuffle 之后的新位置
                token_position = torch.where(shuffle_indices == token_position)[0].item()

            if isinstance(token_position, list):
                print("new value: ", x[0, token_position[0], 0], x[0, token_position[1], 0])
            else:
                print("new value: ", x[0, token_position, 0])
            print("new token_position: ", token_position)

            if self.save is not None and indexes is not None:
                # 只有在需要保存时才计算 last_map
                for i, blk in enumerate(self.layers):
                    if i == len(self.layers) - 1:
                        # 计算规范化值和 qkv 分解
                        y = blk.norm1(hidden_states)
                        B, N, C = y.shape
                        qkv = blk.mixer.attn.qkv(y).reshape(B, N, 3, blk.mixer.attn.num_heads, C // blk.mixer.attn.num_heads).permute(2, 0, 3, 1, 4)
                        q, k, v = qkv[0], qkv[1], qkv[2]

                        # 计算注意力矩阵并进行 softmax
                        att = (q @ k.transpose(-2, -1)) * blk.mixer.attn.scale
                        att = att.softmax(dim=-1)

                        # 生成 last_map
                        last_map = (att[:, :, :2, 2:].detach().cpu().numpy()).sum(axis=1).sum(axis=1)
                        last_map = last_map.reshape([last_map.shape[0], self.patch_embed.grid_size[0], self.patch_embed.grid_size[1]])

                # 保存 last_map 图像
                for j, index in enumerate(indexes.cpu().numpy()):
                    plt.imsave(
                        os.path.join(self.save, str(indexes[j].cpu().numpy()) + '.png'),
                        np.tile(np.expand_dims(last_map[j] / np.max(last_map[j]), 2), [1, 1, 3])  # 可视化和保存
                    )
                pass


        if_flip_img_sequences = False
        if self.flip_img_sequences_ratio > 0 and (self.flip_img_sequences_ratio - random.random()) > 1e-5:
            x = x.flip([1])
            if_flip_img_sequences = True

        # mamba impl
        residual = None
        hidden_states = x
        if not self.if_bidirectional:
            for layer in self.layers:

                if if_flip_img_sequences and self.if_rope:
                    hidden_states = hidden_states.flip([1])
                    if residual is not None:
                        residual = residual.flip([1])

                # rope about
                if self.if_rope:
                    hidden_states = self.rope(hidden_states)
                    if residual is not None and self.if_rope_residual:
                        residual = self.rope(residual)

                if if_flip_img_sequences and self.if_rope:
                    hidden_states = hidden_states.flip([1])
                    if residual is not None:
                        residual = residual.flip([1])

                hidden_states, residual = layer(
                    hidden_states, residual, inference_params=inference_params
                )
        else:
            # get two layers in a single for-loop
            for i in range(len(self.layers) // 2):
                if self.if_rope:
                    hidden_states = self.rope(hidden_states)
                    if residual is not None and self.if_rope_residual:
                        residual = self.rope(residual)

                hidden_states_f, residual_f = self.layers[i * 2](
                    hidden_states, residual, inference_params=inference_params
                )
                hidden_states_b, residual_b = self.layers[i * 2 + 1](
                    hidden_states.flip([1]), None if residual == None else residual.flip([1]), inference_params=inference_params
                )
                hidden_states = hidden_states_f + hidden_states_b.flip([1])
                residual = residual_f + residual_b.flip([1])

        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        # return only cls token if it exists
        if self.if_cls_token:
            if self.use_double_cls_token:
                return (hidden_states[:, token_position[0], :] + hidden_states[:, token_position[1], :]) / 2
            else:
                if self.use_middle_cls_token:
                    return hidden_states[:, token_position, :]
                elif if_random_cls_token_position:
                    return hidden_states[:, token_position, :]
                else:
                    return hidden_states[:, token_position, :]

        if self.final_pool_type == 'none':
            return hidden_states[:, -1, :]
        elif self.final_pool_type == 'mean':
            return hidden_states.mean(dim=1)
        elif self.final_pool_type == 'max':
            return hidden_states
        elif self.final_pool_type == 'all':
            # print(hidden_states.shape)
            B, N, C = hidden_states.shape  # B: Batch size, N: Sequence length, C: Channel dimension
            # if H == W:
            #     hidden_states = self.apply_mapping(hidden_states)
            # print(hidden_states.shape)
            group_size = 16
            assert C % group_size == 0, "The channel dimension must be divisible by the group size."

            # Reshape hidden_states to [B, N, C // group_size, group_size]
            hidden_states = hidden_states.view(B, N, C // group_size, group_size)

            # Perform mean pooling over the last dimension (group_size)
            hidden_states = hidden_states.mean(dim=-1)  # Shape: [B, N, C // group_size]
            # hidden_states = self.norm_f_final(hidden_states)
            # Merge the last two dimensions (N and C // group_size)
            hidden_states = hidden_states.view(B, -1)  # Shape: [B, N * (C // group_size)]

            
            
            return hidden_states
        else:
            raise NotImplementedError

    def forward(self, x, atten=None, indexes=None, return_features=True, inference_params=None, if_random_cls_token_position=False, if_random_token_rank=False):
        x = self.forward_features(x, atten=atten, indexes=indexes, inference_params=inference_params, if_random_cls_token_position=if_random_cls_token_position, if_random_token_rank=if_random_token_rank)
        if return_features:
            return x
        # x = self.head(x)
        if self.final_pool_type == 'max':
            x = x.max(dim=1)[0]
        return x

def resize_positional_embedding(checkpoint, model, img_size):
    """
    Resize positional embedding from the pretrained checkpoint to fit the new model's image size.

    Args:
        checkpoint (dict): The checkpoint dictionary containing 'pos_embed'.
        model (nn.Module): The Vision Transformer model to which the positional embedding belongs.
        img_size (tuple): The new image size (height, width).

    Returns:
        dict: The updated checkpoint with resized 'pos_embed'.
    """
    # Extract positional embedding weights
    weight = checkpoint["model"]['pos_embed']  # Shape: [1, 197, embed_dim]
    ori_size = (14, 14)  # Pre-trained grid size (14x14 for 224x224 input with 16x16 patch size)

    # Calculate new grid size based on input image and patch size
    new_size = (
        img_size[0] // model.patch_embed.patch_size[0],
        img_size[1] // model.patch_embed.patch_size[1]
    )

    # Reshape the positional embedding weights (ignoring cls_token)
    matrix = weight[:, 1:, :].reshape(1, ori_size[0], ori_size[1], -1).permute(0, 3, 1, 2)

    # Resize the positional embedding grid to match the new size
    resize = torchvision.transforms.Resize(new_size, antialias=True)
    new_matrix = resize(matrix).permute(0, 2, 3, 1).reshape(1, -1, weight.shape[-1])

    # Concatenate cls_token back to the resized positional embeddings
    checkpoint["model"]['pos_embed'] = torch.cat([weight[:, :1, :], new_matrix], dim=1)
    return checkpoint
def remove_middle_cls_token(state_dict, model):
    """
    根据模型逻辑从 pos_embed 中去掉 cls_token，并确保其他参数对齐。
    """
    new_state_dict = {}

    for key, value in state_dict.items():
        if "cls_token" in key or "head" in key:
            # print(f"Removing {key} from state_dict")
            continue
        
        # 对 pos_embed 进行调整
        if "pos_embed" in key:
            # print(f"Adjusting positional embedding: {key}")
            # 获取模型中 pos_embed 的形状
            current_pos_embed = model.pos_embed.shape
            B, N, C = current_pos_embed  # [batch_size, num_patches + cls_token, embed_dim]
            
            # 从 checkpoint 中移除中间的 cls_token
            checkpoint_pos_embed = value
            _, checkpoint_N, _ = checkpoint_pos_embed.shape  # [1, num_patches + cls_token, embed_dim]
            
            # 计算中间插入 cls_token 的位置
            token_position = checkpoint_N // 2  # 中间位置
            # 删除 cls_token，保留其余部分
            adjusted_pos_embed = torch.cat((checkpoint_pos_embed[:, :token_position, :],
                                            checkpoint_pos_embed[:, token_position + 1:, :]), dim=1)
            
            # 确保调整后的 pos_embed 与当前模型的 pos_embed 形状匹配
            if adjusted_pos_embed.shape == current_pos_embed:
                new_state_dict[key] = adjusted_pos_embed
            else:
                raise RuntimeError(f"Adjusted pos_embed shape {adjusted_pos_embed.shape} does not match model shape {current_pos_embed}")
        else:
            # 保留其他参数
            new_state_dict[key] = value
    
    return new_state_dict


@register_model
def vim_small_midclstok(pretrained=True, checkpoint_path="./Vim-small-midclstok/vim_s_midclstok_80p5acc.pth", img_size=(224, 224), **kwargs):
    """
    注册模型并支持不同的输入图片大小，动态调整位置嵌入。
    """
    # 使用传入的 img_size 初始化模型
    model = VisionMamba(
        img_size=img_size,  # 从参数中获取 img_size
        patch_size=16, embed_dim=384, depth=24, rms_norm=True, residual_in_fp32=True, fused_add_norm=True,
        final_pool_type='all', if_abs_pos_embed=True, if_rope=False, if_rope_residual=False, bimamba_type="v2",
        if_cls_token=False, if_divide_out=True, use_middle_cls_token=True, **kwargs
    )
    model.default_cfg = _cfg()

    if pretrained:
        # 加载预训练权重
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if "model" in checkpoint:
            # 动态调整位置嵌入
            checkpoint = resize_positional_embedding(checkpoint, model, img_size)
            # 调整 state_dict
            checkpoint["model"] = remove_middle_cls_token(checkpoint["model"], model)
            model.load_state_dict(checkpoint["model"],strict=False)

        else:
            model.load_state_dict(checkpoint)

    return model

