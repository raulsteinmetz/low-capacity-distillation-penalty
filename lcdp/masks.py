'''
    Multi-block mask collator from the official I-JEPA code with two fixes for small patch grids.

    1. Blocks can touch the last row and column. The official code samples the corner with
       randint(0, height - h) and caps h below height, so on an 8x8 grid the last row and column
       are never context or target.
    2. Context masks are cut to the batch minimum by a random subset of each mask rather than by
       its first indices, which in the official code keeps only the top rows.
'''

import math

import torch

from lcdp.ijepa import MaskCollator


class SmallGridMaskCollator(MaskCollator):
    '''I-JEPA multi-block masks where blocks cover the whole grid and context cuts are unbiased.'''

    def _sample_block_size(self, generator, scale, aspect_ratio_scale):
        '''Sample a block height and width that fit the grid, up to its full size.'''
        _rand = torch.rand(1, generator=generator).item()
        min_s, max_s = scale
        mask_scale = min_s + _rand * (max_s - min_s)
        max_keep = int(self.height * self.width * mask_scale)
        min_ar, max_ar = aspect_ratio_scale
        aspect_ratio = min_ar + _rand * (max_ar - min_ar)
        h = min(int(round(math.sqrt(max_keep * aspect_ratio))), self.height)
        w = min(int(round(math.sqrt(max_keep / aspect_ratio))), self.width)
        return (h, w)

    def _sample_block_mask(self, b_size, acceptable_regions=None):
        '''Sample one block anywhere on the grid, optionally restricted to acceptable regions.'''
        h, w = b_size
        tries = 0
        timeout = og_timeout = 20
        while True:
            top = torch.randint(0, self.height - h + 1, (1,))
            left = torch.randint(0, self.width - w + 1, (1,))
            mask = torch.zeros((self.height, self.width), dtype=torch.int32)
            mask[top:top+h, left:left+w] = 1
            if acceptable_regions is not None:
                for k in range(max(len(acceptable_regions) - tries, 0)):
                    mask *= acceptable_regions[k]
            mask = torch.nonzero(mask.flatten())
            if len(mask) > self.min_keep:
                break
            timeout -= 1
            if timeout == 0:
                tries += 1
                timeout = og_timeout
        mask_complement = torch.ones((self.height, self.width), dtype=torch.int32)
        mask_complement[top:top+h, left:left+w] = 0
        return mask.squeeze(), mask_complement

    def __call__(self, batch):
        '''Collate images and sample one set of context and target masks per image.'''
        collated_batch = torch.utils.data.default_collate(batch)
        g = torch.Generator()
        g.manual_seed(self.step())
        p_size = self._sample_block_size(generator=g, scale=self.pred_mask_scale, aspect_ratio_scale=self.aspect_ratio)
        e_size = self._sample_block_size(generator=g, scale=self.enc_mask_scale, aspect_ratio_scale=(1., 1.))

        masks_pred, masks_enc = [], []
        for _ in range(len(batch)):
            pred, complements = [], []
            for _ in range(self.npred):
                mask, mask_c = self._sample_block_mask(p_size)
                pred.append(mask)
                complements.append(mask_c)
            masks_pred.append(pred)
            regions = None if self.allow_overlap else complements
            masks_enc.append([self._sample_block_mask(e_size, acceptable_regions=regions)[0] for _ in range(self.nenc)])

        min_pred = min(len(m) for ms in masks_pred for m in ms)
        min_enc = min(len(m) for ms in masks_enc for m in ms)
        masks_pred = [[m[torch.randperm(len(m))[:min_pred]].sort().values for m in ms] for ms in masks_pred]
        masks_enc = [[m[torch.randperm(len(m))[:min_enc]].sort().values for m in ms] for ms in masks_enc]
        return collated_batch, torch.utils.data.default_collate(masks_enc), torch.utils.data.default_collate(masks_pred)


def build_mask_collator(cfg: dict, img_size: int, patch_size: int) -> SmallGridMaskCollator:
    '''Build the multi-block mask collator.'''
    return SmallGridMaskCollator(
        input_size=(img_size, img_size),
        patch_size=patch_size,
        enc_mask_scale=tuple(cfg['enc_mask_scale']),
        pred_mask_scale=tuple(cfg['pred_mask_scale']),
        aspect_ratio=tuple(cfg['aspect_ratio']),
        nenc=cfg['num_enc_masks'],
        npred=cfg['num_pred_masks'],
        min_keep=cfg['min_keep'],
        allow_overlap=cfg['allow_overlap'],
    )
