from functools import partial
import torch
from dinov2.utils.config import setup
from dinov2.data import (
    collate_data_and_cast,
    DataAugmentationDINO,
    MaskingGenerator)
from dinov2.train.train import build_schedulers

from .scheduler import cosine_lr, const_lr, const_lr_cooldown

def adapted_dino_collate_fn(_dino_collate_fn):
    def dino_collate_fn(x):
        images = torch.stack([s[0] for s in x])
        texts = torch.stack([s[1] for s in x])
        return (images, texts, _dino_collate_fn([(s[2], ()) for s in x])) # (s[2], ()) taken from dinov2
    return dino_collate_fn

def get_dino_data_transforms(dino_cfg):
    dino_data_transform = DataAugmentationDINO(
        dino_cfg.crops.global_crops_scale,
        dino_cfg.crops.local_crops_scale,
        dino_cfg.crops.local_crops_number,
        global_crops_size=dino_cfg.crops.global_crops_size,
        local_crops_size=dino_cfg.crops.local_crops_size,
    )
    inputs_dtype = torch.half
    img_size = dino_cfg.crops.global_crops_size
    patch_size = dino_cfg.student.patch_size
    n_tokens = (img_size // patch_size) ** 2
    mask_generator = MaskingGenerator(
        input_size=(img_size // patch_size, img_size // patch_size),
        max_num_patches=0.5 * img_size // patch_size * img_size // patch_size,
    )
    _dino_collate_fn = partial(
        collate_data_and_cast,
        mask_ratio_tuple=dino_cfg.ibot.mask_ratio_min_max,
        mask_probability=dino_cfg.ibot.mask_sample_probability,
        n_tokens=n_tokens,
        mask_generator=mask_generator,
        dtype=inputs_dtype,
    )
    dino_collate_fn = adapted_dino_collate_fn(_dino_collate_fn)
    return dino_data_transform, dino_collate_fn

def build_schedulers(dino_cfg, warmup_length, steps):
    teacher_cfg = dino_cfg.teacher
    if teacher_cfg.teacher_temp_type == "constant":
        teacher_temp_scheduler = cosine_lr(
            optimizer=None, base_lr=teacher_cfg.teacher_temp, warmup_length=warmup_length, steps=steps)
    elif teacher_cfg.teacher_temp_type == "constant":
        teacher_temp_scheduler = const_lr(
            optimizer=None, base_lr=teacher_cfg.teacher_temp, warmup_length=warmup_length, steps=steps)
    if teacher_cfg.momentum_type == "cosine":
        teacher_momentum_scheduler = cosine_lr(
            optimizer=None, base_lr=teacher_cfg.momentum_teacher, warmup_length=warmup_length, steps=steps)
    elif teacher_cfg.momentum_type == "constant":
        teacher_momentum_scheduler = const_lr(
            optimizer=None, base_lr=teacher_cfg.momentum_teacher, warmup_length=warmup_length, steps=steps)
    optim_cfg = dino_cfg.optim
    wd_scheduler = cosine_lr(
        optimizer=None, base_lr=optim_cfg.weight_decay, warmup_length=warmup_length, steps=steps)

    return {
        "teacher_temp_scheduler": teacher_temp_scheduler,
        "teacher_momentum_scheduler": teacher_momentum_scheduler,
        "wd_scheduler": wd_scheduler,
    }