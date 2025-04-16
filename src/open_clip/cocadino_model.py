""" CLIP Model

Adapted from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
"""
import copy
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .transformer import text_global_pool, AttentionalPooler, LayerNorm
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
            d_model=embed_dim,
            context_dim=vision_cfg.width,
            n_head=vision_cfg.attn_pooler_heads,
        )
        self.attn_pool_norm = LayerNorm(embed_dim)
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
        self.transformer.grad_checkpointing = enable

    def _encode_image(self, image, normalize: bool = False, teacher_temp: Optional[float] = None):
        if teacher_temp is None:
            teacher_temp = 1.0
        dino_loss_dict, features_dict = self.visual.forward_backward(image, teacher_temp)
        features = features_dict["x_norm"]
        features = self.attn_pooler(features)
        features = self.attn_pool_norm(features)
        return dino_loss_dict, (F.normalize(features, dim=-1) if normalize else features)

    def _encode_text(self, text, normalize: bool = True):
        text_latent, token_emb = self.text(text)
        text_latent = F.normalize(text_latent, dim=-1) if normalize else text_latent
        return text_latent, token_emb

    def encode_image(self, image, normalize: bool = False):
        features = self.visual.student.backbone(image)
        features = self.attn_pooler(features)
        features = self.attn_pool_norm(features)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize: bool = True):
        text_latent, _ = self._encode_text(text, normalize=normalize)
        return text_latent

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
            output_labels: bool = True,
    ):

        dino_loss_dict, image_features = self._encode_image(
            image, normalize=True, teacher_temp=teacher_temp)

        image_latent, image_embs = image_features[:, 0], image_features[:, 1:]

        # text_features = self.encode_text(text, normalize=True) if text is not None else None
        text_latent, token_embs = self._encode_text(text)

        labels: Optional[torch.Tensor] = text[:, 1:] if output_labels else None
        if output_labels:
            # align text_embs and thus logits with labels for teacher-forcing caption loss
            token_embs = token_embs[:, :-1]

        logits = self.text_decoder(image_embs[:token_embs.shape[0], :], token_embs)

        if self.output_dict:
            out_dict = {
                "image_features": image_latent,
                "text_features": text_latent,
                "logits": logits,
                "logit_scale": self.logit_scale.exp(),
                "labels": labels,
            }
            if dino_loss_dict is not None:
                out_dict['dino_loss'] = dino_loss_dict
            if self.logit_bias is not None:
                out_dict['logit_bias'] = self.logit_bias
            return out_dict

        if self.logit_bias is not None:
            return image_features, text_features, self.logit_scale.exp(), self.logit_bias
        return image_features, text_features, self.logit_scale.exp()
