from .model import CLIP

class SelfSustainCLIP(CLIP):
    output_dict: torch.jit.Final[bool]

    def __init__(
            self,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            quick_gelu: bool = False,
            cast_dtype: Optional[torch.dtype] = None,
            output_dict: bool = False,
    ):
        super().__init__()
        self.im2im_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.txt2txt_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.oracle_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        
        def forward(self, image, text):
            output = super().forward(image, text)
            output["im2im_logit_scale"] = self.im2im_logit_scale.exp()
            output["txt2txt_logit_scale"] = self.txt2txt_logit_scale.exp()
            output["oracle_logit_scale"] = self.oracle_logit_scale.exp()
            return output