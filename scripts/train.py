'''
    Train a JEPA on CIFAR-10 with optional anti-collapse terms and log collapse and probe metrics.
'''

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from lcdp.data import eval_loader, pretrain_loader, tinted_loader
from lcdp.ijepa import build_encoder, build_predictor, select_targets
from lcdp.masks import build_mask_collator
from lcdp.metrics import effective_rank, linear_decodability, linear_probe, token_metrics
from lcdp.paths import RUNS, ROOT
from lcdp.regularizers import WeakReproducer, build_sigreg, target_norm, vicreg_var_cov, weak_penalty

REC_KEYS = ['loss', 'pred', 'weak', 'weak_fit', 'vic_std', 'vic_cov', 'sigreg', 'pooled_std', 'grad_norm']


def load_config(path: Path, overrides: list[str]) -> dict:
    '''Load a yaml config and apply dotted key=value overrides.'''
    cfg = yaml.safe_load(path.read_text())
    for item in overrides:
        key, value = item.split('=', 1)
        node = cfg
        *parents, leaf = key.split('.')
        for p in parents:
            node = node[p]
        node[leaf] = yaml.safe_load(value)
    return cfg


def param_groups(modules: list) -> list[dict]:
    '''Split params into a weight-decayed group and a group of biases and 1-D params without decay.'''
    decay, no_decay = [], []
    for m in modules:
        for n, p in m.named_parameters():
            if not p.requires_grad:
                continue
            (no_decay if n.endswith('bias') or p.ndim == 1 else decay).append(p)
    return [{'params': decay}, {'params': no_decay, 'weight_decay': 0.0}]


def lr_at(step: int, total: int, warmup: int, cfg: dict) -> float:
    '''Linear warmup then cosine decay.'''
    if step < warmup:
        return cfg['lr'] * (step + 1) / warmup
    progress = (step - warmup) / (total - warmup)
    return cfg['final_lr'] + 0.5 * (cfg['lr'] - cfg['final_lr']) * (1 + math.cos(math.pi * progress))


def sample_tokens(num_tokens: int, cfg: dict, device: torch.device) -> torch.Tensor:
    '''Random token indices used by the token-level weak penalty.'''
    return torch.randperm(num_tokens, device=device)[:cfg['token_sample']]


@torch.no_grad()
def extract(encoder, weak, loader, device, amp, cfg) -> dict:
    '''Pooled embeddings, labels, tokens of the first images, and the mean per-batch weak penalty over a split.'''
    pooled, labels, tokens, scores = [], [], [], []
    seen = 0
    for imgs, y in loader:
        imgs = imgs.to(device, non_blocking=True)
        with amp:
            h = encoder(imgs).float()
            w_out = weak(imgs).float()
        pooled.append(h.mean(1))
        labels.append(y.to(device))
        if seen < cfg['eval']['token_images']:
            tokens.append(h[:cfg['eval']['token_images'] - seen])
            seen += len(tokens[-1])
        idx = sample_tokens(h.size(0) * h.size(1), cfg['weak_reproducer'], device)
        scores.append(weak_penalty(w_out, target_norm(h), cfg['weak_reproducer'], idx).item())
    return {'pooled': torch.cat(pooled), 'labels': torch.cat(labels), 'tokens': torch.cat(tokens),
            'weak_score': sum(scores) / len(scores)}


@torch.no_grad()
def tint_decodability(encoder, loader, device, amp, cfg) -> float:
    '''How well a linear map recovers the tint color of tinted test images from pooled embeddings, as R2.'''
    pooled, colors = [], []
    for imgs, c in loader:
        with amp:
            pooled.append(encoder(imgs.to(device, non_blocking=True)).float().mean(1))
        colors.append(c.to(device))
    return linear_decodability(torch.cat(pooled), torch.cat(colors), cfg['ridge'])


def evaluate(encoder, weak, loaders, device, amp, cfg) -> dict:
    '''Collapse metrics, weak-model dependence, tint decodability and linear-probe accuracy.'''
    encoder.eval()
    weak.eval()
    train = extract(encoder, weak, loaders['train'], device, amp, cfg)
    test = extract(encoder, weak, loaders['test'], device, amp, cfg)
    tint_r2 = tint_decodability(encoder, loaders['tinted'], device, amp, cfg['eval']['tint'])
    encoder.train()
    weak.train()
    std = test['pooled'].std(0)
    rank = effective_rank(test['pooled'])
    return {
        'erank_pooled': rank['erank'],
        'entropy_pooled': rank['entropy'],
        **token_metrics(test['tokens']),
        'std_mean': std.mean().item(),
        'std_min': std.min().item(),
        'std_median': std.median().item(),
        'std_per_dim': std.tolist(),
        'weak_score_test': test['weak_score'],
        'tint_r2': tint_r2,
        'probe_acc': linear_probe(train['pooled'], train['labels'], test['pooled'], test['labels'], cfg['eval']['probe']),
    }


def main() -> None:
    '''Train one run and write its config, logs, metrics and checkpoint to runs/<name>.'''
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, default=ROOT / 'configs' / 'train' / 'base.yaml')
    ap.add_argument('--name', required=True, help='Run directory name under runs/.')
    ap.add_argument('--set', nargs='*', default=[], help='Overrides as dotted.key=value.')
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    out = RUNS / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(yaml.safe_dump(cfg, sort_keys=False))

    torch.manual_seed(cfg['seed'])
    device = torch.device(cfg['device'])
    amp = torch.autocast(device.type, dtype=torch.bfloat16, enabled=cfg['bf16'])

    collator = build_mask_collator(cfg['mask'], cfg['model']['img_size'], cfg['model']['patch_size'])
    loader = pretrain_loader(cfg['data'], collator)
    eval_loaders = {'train': eval_loader(True, cfg['data']), 'test': eval_loader(False, cfg['data']),
                    'tinted': tinted_loader(cfg['eval']['tint'], cfg['data'])}

    encoder = build_encoder(cfg['model']).to(device)
    predictor = build_predictor(cfg['model']).to(device)
    wcfg = cfg['weak_reproducer']
    weak = WeakReproducer(wcfg, cfg['model']['img_size'], cfg['model']['patch_size'], cfg['model']['embed_dim']).to(device)
    sigreg = build_sigreg(cfg['sigreg']).to(device)
    print(f'params E {sum(p.numel() for p in encoder.parameters()) / 1e6:.3f}M '
          f'P {sum(p.numel() for p in predictor.parameters()) / 1e6:.3f}M '
          f'W {sum(p.numel() for p in weak.parameters() if p.requires_grad) / 1e3:.1f}K')

    ocfg = cfg['optim']
    main_params = list(encoder.parameters()) + list(predictor.parameters())
    opt = torch.optim.AdamW(param_groups([encoder, predictor]), lr=ocfg['lr'], weight_decay=ocfg['weight_decay'])
    opt_w = torch.optim.AdamW([p for p in weak.parameters() if p.requires_grad], lr=wcfg['lr'], weight_decay=wcfg['weight_decay'])
    steps_per_epoch = len(loader) if ocfg['max_steps_per_epoch'] is None else ocfg['max_steps_per_epoch']
    total = ocfg['epochs'] * steps_per_epoch
    warmup = ocfg['warmup_epochs'] * steps_per_epoch
    nenc = cfg['mask']['num_enc_masks']

    step = 0
    t0 = time.time()
    with open(out / 'train.jsonl', 'w') as train_log, open(out / 'metrics.jsonl', 'w') as metrics_log:
        m = {'epoch': 0, 'step': 0, **evaluate(encoder, weak, eval_loaders, device, amp, cfg)} # random-init reference
        metrics_log.write(json.dumps(m) + '\n')
        print(f'eval epoch 0 ' + ' '.join(f'{k} {v:.4f}' for k, v in m.items() if not isinstance(v, list)), flush=True)
        for epoch in range(ocfg['epochs']):
            sums = None
            for i, ((imgs, _), masks_enc, masks_pred) in enumerate(loader):
                if i == steps_per_epoch:
                    break
                imgs = imgs.to(device, non_blocking=True)
                masks_enc = [m.to(device, non_blocking=True) for m in masks_enc]
                masks_pred = [m.to(device, non_blocking=True) for m in masks_pred]

                with amp:
                    h = encoder(imgs)
                    z = predictor(encoder(imgs, masks_enc), masks_enc, masks_pred)
                hn = target_norm(h.float())
                idx = sample_tokens(hn.size(0) * hn.size(1), wcfg, device)

                # W inner steps maximize the penalty against the current E, which they cannot change
                for _ in range(wcfg['inner_steps']):
                    with amp:
                        w_fit = weak(imgs)
                    loss_w = -weak_penalty(w_fit.float(), hn.detach(), wcfg, idx)
                    opt_w.zero_grad()
                    loss_w.backward()
                    opt_w.step()
                with torch.no_grad(), amp:
                    w_out = weak(imgs)

                # E and P step, W is frozen so the weak penalty only moves E
                for g in opt.param_groups:
                    g['lr'] = lr_at(step, total, warmup, ocfg)
                t = select_targets(hn, masks_pred, nenc)
                if cfg['stop_grad']:
                    t = t.detach()
                pred_loss = F.mse_loss(z.float(), t)
                weak_loss = weak_penalty(w_out.float(), hn, wcfg, idx)
                pooled = hn.mean(1)
                std_loss, cov_loss = vicreg_var_cov(pooled)
                sig_loss = sigreg(pooled)
                loss = (pred_loss + wcfg['weight'] * weak_loss + cfg['vicreg']['std_weight'] * std_loss
                        + cfg['vicreg']['cov_weight'] * cov_loss + cfg['sigreg']['weight'] * sig_loss)
                opt.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(main_params, ocfg['grad_clip'])
                opt.step()

                rec = torch.stack([loss, pred_loss, weak_loss, -loss_w, std_loss, cov_loss, sig_loss,
                                   pooled.std(0).mean(), grad_norm]).detach()
                sums = rec if sums is None else sums + rec
                if step % 50 == 0:
                    train_log.write(json.dumps({'step': step, 'epoch': epoch, 'lr': opt.param_groups[0]['lr'],
                                                **dict(zip(REC_KEYS, rec.tolist()))}) + '\n')
                    train_log.flush()
                step += 1

            avg = dict(zip(REC_KEYS, (sums / steps_per_epoch).tolist()))
            print(f'epoch {epoch} step {step} {time.time() - t0:.0f}s ' + ' '.join(f'{k} {v:.4f}' for k, v in avg.items()), flush=True)
            if (epoch + 1) % cfg['eval']['every_epochs'] == 0 or epoch + 1 == ocfg['epochs']:
                m = evaluate(encoder, weak, eval_loaders, device, amp, cfg)
                m = {'epoch': epoch + 1, 'step': step, **{f'train_{k}': v for k, v in avg.items()}, **m}
                metrics_log.write(json.dumps(m) + '\n')
                metrics_log.flush()
                print(f'eval epoch {epoch + 1} ' + ' '.join(f'{k} {v:.4f}' for k, v in m.items() if not isinstance(v, list)), flush=True)

    torch.save({'encoder': encoder.state_dict(), 'predictor': predictor.state_dict(), 'weak': weak.state_dict()},
               out / 'checkpoint.pt')


if __name__ == '__main__':
    main()
