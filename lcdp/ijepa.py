'''
    Builds the encoder and predictor and exposes the mask collator from the official I-JEPA code in external/ijepa.
'''

import sys
from functools import partial

import torch

from lcdp.paths import IJEPA

sys.path.insert(0, str(IJEPA)) # the official code imports its own top-level package named src
from src.masks.multiblock import MaskCollator
from src.masks.utils import apply_masks
from src.models.vision_transformer import VisionTransformer, vit_predictor
from src.utils.tensors import repeat_interleave_batch


def build_encoder(cfg: dict) -> VisionTransformer:
    '''Build the context and target encoder E.'''
    return VisionTransformer(
        img_size=[cfg['img_size']],
        patch_size=cfg['patch_size'],
        embed_dim=cfg['embed_dim'],
        depth=cfg['depth'],
        num_heads=cfg['num_heads'],
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
    )


def build_predictor(cfg: dict) -> torch.nn.Module:
    '''Build the predictor P that maps context tokens to target-block embeddings.'''
    return vit_predictor(
        num_patches=(cfg['img_size'] // cfg['patch_size']) ** 2,
        embed_dim=cfg['embed_dim'],
        predictor_embed_dim=cfg['pred_embed_dim'],
        depth=cfg['pred_depth'],
        num_heads=cfg['pred_num_heads'],
    )


def select_targets(h: torch.Tensor, masks_pred: list, num_enc_masks: int) -> torch.Tensor:
    '''Gather target-block tokens from full-image embeddings, in the order the predictor outputs them.'''
    return repeat_interleave_batch(apply_masks(h, masks_pred), len(h), repeat=num_enc_masks)
