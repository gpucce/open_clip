import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None
    
try:
    from transformers import AutoModel, AutoTokenizer
except ImportError:
    transformers = None

def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features


class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T
        
        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features, logit_scale, output_dict=False):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss

class SelfSustainClipLoss(ClipLoss):
    
    def __init__(
        self,
        local_loss=False,
        gather_with_grad=False,
        cache_labels=False,
        rank=0,
        world_size=1,
        use_horovod=False,
        oracle_name_or_path"sentence-transformers/all-MiniLM-L6-v2"
        lambda_start_epoch=0
        lambda_end_epoch=1
    ):
        super().__init__()
        
        if transformers is None:
            raise RuntimeError("Please `pip install transformers` to use SelfSustainClipLoss")

        self.text_oracle = transformers.AutoModel.from_pretrained(oracle_name_or_path)
        self.oracle_tokenizer = transformers.AutoTokenizer.from_pretrained(oracle_name_or_path)
        self.lambda_start_epoch = lambda_start_epoch
        self.lambda_end_epoch = lambda_end_epoch
        
        
    def tokenize(self, x, device):
        return {
            i:j.to(device) for i,j in self.oracle_tokenizer(
                x,
                return_tensors="pt",
                max_length=256, 
                padding="max_length", 
                truncation=True
            ).items()
        }
    
    def _listnet_loss(self, teacher_scores, student_scores, eps = 1e-10):
        # https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2007-40.pdf
        preds_smax = F.softmax(student_scores, dim=1)
        true_smax = F.softmax(teacher_scores, dim=1)
        preds_smax = preds_smax + eps
        preds_log = torch.log(preds_smax)
        cost = torch.mean(-torch.sum(true_smax * preds_log, dim=1))
        
    def forward(
        self, 
        image_features,
        text_features,
        logit_scale,
        im2im_logit_scale,
        txt2txt_logit_scale,
        oracle_logit_sclae,
        text=None
    ):

        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)
        
        with torch.no_grad():
            if text is None:
                oracle_s_emb = self.text_oracle(input_ids=text_features.to(device))
            else:
                oracle_s_emb = self.text_oracle(self.tokenize(text, device))
        
        # TODO: create all this logit scales
        im2im_logits = self.get_logits(image_features, image_features, im2im_logit_scale)
        txt2txt_logits = self.get_logits(text_features, text_features, txt2txt_logit_scale)
        oracle_logits = self.get_logits(oracle_s_emb, oracle_s_emb, oracle_logit_scale)

        im2im_vs_teacher_loss = self._listnet_loss(oracle_logits, im2im_logits)
        txt2txt_vs_teacher_loss = self._listnet_loss(oracle_logits, txt2txt_logits)

        # listnet losses against learned multimodal embeddings
        # 1.symmetric listness loss for text
        multimod_embs_vs_txt2txt = (
            self.listnet_loss(txt2txt_logits, logits_per_image) + 
            self.listnet_loss(txt2txt_logits, logits_per_text) +
            self.listnet_loss(logits_per_image, txt2txt_logits) + 
            self.listnet_loss(logits_per_text, txt2txt_logits)
        ) / 4
        # 2.symmetric listness loss for motion
        multimod_embs_vs_im2im = (
            self.listnet_loss(im2im_logits, logits_per_image) + 
            self.listnet_loss(im2im_logits, logits_per_text) + 
            self.listnet_loss(logits_per_image, im2im_logits) + 
            self.listnet_loss(logits_per_text, im2im_logits)
        ) / 4

        # compute bidirectional CE loss (standard cross-modal CLIP objective)
        labels = self.get_ground_truth(device, logits_per_image.shape[0])
        clip_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        # compute lambda as function of epoch
        # linear swipe between lambda_start_epoch and lambda_end_epoch
        lamb = (
            (torch.Tensor([epoch]).to(im.device) - self.lambda_start_epoch) / 
            (self.lambda_end_epoch - self.lambda_start_epoch)
        )
        # clamp between 0 and 1
        lamb = lamb.clamp(0, 1)
        
        self_sustain_loss = lamb * (multimod_embs_vs_im2im + multimod_embs_vs_txt2txt)
        oracle_loss = (1 - lamb) * (txt2txt_vs_teacher_loss + m2m_vs_teacher_loss)
        
        return {
            "contrastive_loss": clip_loss,
            'self_sustain_loss': self_sustain_loss,
            'oracle_loss': oracle_loss,
        }
        
        
        
class CoCaLoss(ClipLoss):
    def __init__(
            self,
            caption_loss_weight,
            clip_loss_weight,
            pad_id=0,  # pad_token for open_clip custom tokenizer
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__(
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            rank=rank,
            world_size=world_size,
            use_horovod=use_horovod
        )

        self.clip_loss_weight = clip_loss_weight
        self.caption_loss_weight = caption_loss_weight
        self.caption_loss = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, image_features, text_features, logits, labels, logit_scale, output_dict=False):
        clip_loss = super().forward(image_features, text_features, logit_scale)
        clip_loss = self.clip_loss_weight * clip_loss

        caption_loss = self.caption_loss(
            logits.permute(0, 2, 1),
            labels,
        )
        caption_loss = caption_loss * self.caption_loss_weight

        if output_dict:
            return {"contrastive_loss": clip_loss, "caption_loss": caption_loss}

        return clip_loss, caption_loss


class DistillClipLoss(ClipLoss):

    def dist_loss(self, teacher_logits, student_logits):
        return -(teacher_logits.softmax(dim=1) * student_logits.log_softmax(dim=1)).sum(dim=1).mean(dim=0)

    def forward(
            self,with torch.no_grad():
            oracle_s_emb = self.text_oracle_model.encode(texts, convert_to_tensor=True, normalize_embeddings=True).to(im.device)
            image_features,
            text_features,
            logit_scale,
            dist_image_features,
            dist_text_features,
            dist_logit_scale,
            output_dict=False,
    ):
        logits_per_image, logits_per_text = \
            self.get_logits(image_features, text_features, logit_scale)

        dist_logits_per_image, dist_logits_per_text = \
            self.get_logits(dist_image_features, dist_text_features, dist_logit_scale)

        labels = self.get_ground_truth(image_features.device, logits_per_image.shape[0])

        contrastive_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        distill_loss = (
            self.dist_loss(dist_logits_per_image, logits_per_image) +
            self.dist_loss(dist_logits_per_text, logits_per_text)
        ) / 2

        if output_dict:
            return {"contrastive_loss": contrastive_loss, "distill_loss": distill_loss}

        return contrastive_loss, distill_loss
        