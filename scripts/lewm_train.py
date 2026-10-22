'''
    Train LeWorldModel (external/le-wm/train.py) with the weak-reproducer penalty added to its loss.
    Run with envs/lewm/bin/python -m scripts.lewm_train from the repo root, with hydra overrides as arguments.
'''

import os
import sys
from functools import partial

from lcdp.paths import LEWM, RUNS, STABLEWM_HOME

os.environ['STABLEWM_HOME'] = str(STABLEWM_HOME)
sys.path.insert(0, str(LEWM)) # le-wm modules jepa, module and utils, also used as hydra _target_ paths

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf, open_dict

from lcdp.regularizers import WeakReproducer, cca_dependence, reproducibility
from module import SIGReg
from utils import SaveCkptCallback, get_column_normalizer, get_img_preprocessor


def weak_dependence(w: torch.Tensor, e: torch.Tensor, cfg) -> torch.Tensor:
    '''Dependence of frame embeddings e (n, D) on pooled W features w (n, D).'''
    if cfg.penalty == 'r2':
        return reproducibility(w, e, cfg.var_floor)
    return cca_dependence(w, e, cfg.cca_ridge)


def lejepa_weak_forward(self, batch, stage, cfg):
    '''LeWM forward of external/le-wm/train.py plus the weak penalty and the W loss.'''
    ctx_len = cfg.history_size
    n_preds = cfg.num_preds
    lambd = cfg.loss.sigreg.weight

    batch['action'] = torch.nan_to_num(batch['action'], 0.0) # NaN at sequence boundaries, as in le-wm

    output = self.model.encode(batch)
    emb = output['emb'] # (B, T, D)
    act_emb = output['act_emb']

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]
    tgt_emb = emb[:, n_preds:]
    pred_emb = self.model.predict(ctx_emb, ctx_act)

    w_out = self.weak(batch['pixels'].flatten(0, 1)).float().mean(1) # (B*T, D)
    e = emb.flatten(0, 1).float()
    weak_e = weak_dependence(w_out.detach(), e, cfg.weak) # moves E only
    weak_w = weak_dependence(w_out, e.detach(), cfg.weak) # moves W only, W maximizes it

    output['pred_loss'] = (pred_emb - tgt_emb).pow(2).mean()
    output['sigreg_loss'] = self.sigreg(emb.transpose(0, 1))
    output['weak_loss'] = weak_e
    output['loss'] = output['pred_loss'] + lambd * output['sigreg_loss'] + cfg.weak.weight * weak_e - weak_w
    output['emb_std'] = e.std(0).mean()

    logged = {f'{stage}/{k}': v.detach() for k, v in output.items() if 'loss' in k or k == 'emb_std'}
    self.log_dict(logged, on_step=stage == 'fit', on_epoch=stage != 'fit', sync_dist=True)
    return output


@hydra.main(version_base=None, config_path='../configs/lewm', config_name='train')
def run(cfg):
    '''Build data, model, W and trainer as external/le-wm/train.py does, and train.'''
    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop('name')
    dataset = swm.data.load_dataset(dataset_name, transform=None, **dataset_cfg)
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith('pixels'):
                continue
            transforms.append(get_column_normalizer(dataset, col, col))
        cfg.model.action_encoder.input_dim = cfg.data.dataset.frameskip * dataset.get_dim('action')
    dataset.transform = spt.data.transforms.Compose(*transforms)

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen)
    train = torch.utils.data.DataLoader(train_set, **cfg.loader, shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)

    torch.manual_seed(cfg.seed)
    world_model = hydra.utils.instantiate(cfg.model)
    patch_size = cfg.model.encoder.patch_size
    weak = WeakReproducer(OmegaConf.to_container(cfg.weak), cfg.img_size, patch_size, cfg.embed_dim)

    optimizers = {
        'model_opt': {
            'modules': 'model',
            'optimizer': dict(cfg.optimizer),
            'scheduler': {'type': 'LinearWarmupCosineAnnealingLR'},
            'interval': 'epoch',
        },
        'weak_opt': {
            'modules': 'weak',
            'optimizer': {'type': 'AdamW', 'lr': cfg.weak.lr, 'weight_decay': cfg.weak.weight_decay},
        },
    }
    module = spt.Module(
        model=world_model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        weak=weak,
        forward=partial(lejepa_weak_forward, cfg=cfg),
        optim=optimizers,
    )

    out = RUNS / cfg.output_model_name
    out.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out / 'config.yaml')
    spt.set(cache_dir=str(out)) # the Manager writes metrics.csv and checkpoints under out/runs/<date>/<time>/<id>
    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[SaveCkptCallback(run_name=cfg.output_model_name, cfg=cfg.model, epoch_interval=1)],
        num_sanity_val_steps=1,
        enable_checkpointing=False,
    )
    spt.Manager(trainer=trainer, module=module, data=spt.data.DataModule(train=train, val=val), ckpt_path=None)()


if __name__ == '__main__':
    run()
