""" CLIP Model

Adapted from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
"""
import copy
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .transformer import text_global_pool, AttentionalPooler
from .model import CLIPVisionCfg, CLIPTextCfg, _build_text_tower
from .coca_model import _build_text_decoder_tower, MultimodalCfg
from .silc_model import _build_dino_vision_tower


class CoCaDino(nn.Module):
    output_dict: torch.jit.Final[bool]

    def __init__(
        self,
        embed_dim: int,
        multimodal_cfg: MultimodalCfg,
        vision_cfg: CLIPVisionCfg,
        text_cfg: CLIPTextCfg,
        quick_gelu: bool = False,
        init_logit_scale: float = np.log(1 / 0.07),
        init_logit_bias: Optional[float] = None,
        nonscalar_logit_scale: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
        output_dict: bool = False,
    ):
        super().__init__()
        self.output_dict = output_dict
        multimodal_cfg = MultimodalCfg(**multimodal_cfg) if isinstance(multimodal_cfg, dict) else multimodal_cfg
        vision_cfg = CLIPVisionCfg(**vision_cfg) if isinstance(vision_cfg, dict) else vision_cfg
        text_cfg = CLIPTextCfg(**text_cfg) if isinstance(text_cfg, dict) else text_cfg

        vocab_size = (
            text_cfg.vocab_size  # for hf models
            if hasattr(text_cfg, "hf_model_name") and text_cfg.hf_model_name is not None
            else text_cfg.vocab_size
        )

        self.visual = _build_dino_vision_tower(vision_cfg)
        self.attn_pooler = AttentionalPooler(

        )
        self.text = _build_text_tower(
            embed_dim=embed_dim,
            text_cfg=text_cfg,
            quick_gelu=quick_gelu,
            cast_dtype=cast_dtype,
        )

        self.text_decoder = _build_text_decoder_tower(
            vocab_size,
            multimodal_cfg=multimodal_cfg,
            quick_gelu=quick_gelu,
            cast_dtype=cast_dtype,
        )

        lshape = [1] if nonscalar_logit_scale else []
        self.logit_scale = nn.Parameter(torch.ones(lshape) * init_logit_scale)

        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones(lshape) * init_logit_bias)
        else:
            self.logit_bias = None

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        if not self.skip_text:
            self.transformer.grad_checkpointing = enable

    def _encode_image(self, image, normalize: bool = False, teacher_temp: Optional[float] = None):
        if teacher_temp is None:
            teacher_temp = 1.0
        dino_loss_dict, features_dict = self.visual.forward_backward(image, teacher_temp)
        features = features_dict["x_norm_clstoken"]
        if not self.skip_text:
            features = features @ self.visual_proj
        return dino_loss_dict, (F.normalize(features, dim=-1) if normalize else features)

    def encode_image(self, image, normalize: bool = False):
        features = self.visual.student.backbone(image)
        features = features @ self.visual_proj
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize: bool = False):
        cast_dtype = self.transformer.get_cast_dtype()

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(cast_dtype)
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        x, _ = text_global_pool(x, text, self.text_pool_type)
        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                x = self.text_projection(x)
            else:
                x = x @ self.text_projection

        return F.normalize(x, dim=-1) if normalize else x

    # def get_logits(self, image, text):
    #     image_features = self.encode_image(image, normalize=True)
    #     text_features = self.encode_text(text, normalize=True)
    #     image_logits = self.logit_scale.exp() * image_features @ text_features.T
    #     if self.logit_bias is not None:
    #         image_logits += self.logit_bias
    #     text_logits = image_logits.T
    #     return image_logits, text_logits

    def forward(
            self,
            image: Optional[torch.Tensor] = None,
            text: Optional[torch.Tensor] = None,
            teacher_temp: Optional[float] = None,
    ):
        dino_loss_dict = None
        if text is not None:
            dino_loss_dict, image_features = self._encode_image(
                image, normalize=True, teacher_temp=teacher_temp)
        else:
            image_features = self.encode_image(image, normalize=True) if image is not None else None

        text_features = None
        if not self.skip_text:
            text_features = self.encode_text(text, normalize=True) if text is not None else None

        if self.output_dict:
            out_dict = {
                "image_features": image_features,
                "text_features": text_features,
                "logit_scale": self.logit_scale.exp() if not self.skip_text else None,
            }
            if dino_loss_dict is not None:
                out_dict['dino_loss'] = dino_loss_dict
            if self.logit_bias is not None:
                out_dict['logit_bias'] = self.logit_bias
            return out_dict

        if self.logit_bias is not None:
            return image_features, text_features, self.logit_scale.exp(), self.logit_bias
        return image_features, text_features, self.logit_scale.exp()
