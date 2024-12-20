from functools import partial
import torch
from dinov2.utils.config import setup
from dinov2.data import (
    collate_data_and_cast,
    DataAugmentationDINO,
    MaskingGenerator)

def adapted_dino_collate_fn(dino_cfg, _dino_collate_fn):
    def dino_collate_fn(x):
        
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