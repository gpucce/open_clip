from typing import Optional
import torch
import numpy as np
from .model import CustomTextCLIP, CLIP, CLIPVisionCfg, CLIPTextCfg

class SelfSustainCLIP(CustomTextCLIP):

    def __init__(
            self,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            quick_gelu: bool = False,
            cast_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(
            embed_dim=embed_dim,
            vision_cfg=vision_cfg,
            text_cfg=text_cfg,
            quick_gelu=quick_gelu,
            cast_dtype=cast_dtype
        )
        self.im2im_logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.txt2txt_logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.oracle_logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        
    def forward(self, image, text):
        output = super().forward(image, text)
        output["im2im_logit_scale"] = self.im2im_logit_scale.exp()
        output["txt2txt_logit_scale"] = self.txt2txt_logit_scale.exp()
        output["oracle_logit_scale"] = self.oracle_logit_scale.exp()
        return output