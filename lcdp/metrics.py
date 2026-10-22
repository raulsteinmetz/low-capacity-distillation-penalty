'''
    Collapse and downstream metrics computed on frozen embeddings.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


def effective_rank(x: torch.Tensor) -> dict:
    '''Entropy of normalized singular values of centered embeddings, and its exponential.'''
    s = torch.linalg.svdvals((x - x.mean(0)).double())
    p = s / s.sum()
    entropy = -(p * torch.log(p + 1e-12)).sum()
    return {'entropy': entropy.item(), 'erank': entropy.exp().item()}


def token_metrics(tokens: torch.Tensor) -> dict:
    '''Rank and spread of tokens (B, N, D), raw and with the per-position mean over images removed.'''
    content = tokens - tokens.mean(0, keepdim=True) # removes what depends on position only
    return {
        'erank_tokens': effective_rank(tokens.flatten(0, 1))['erank'],
        'erank_tokens_content': effective_rank(content.flatten(0, 1))['erank'],
        'token_std_mean': tokens.flatten(0, 1).std(0).mean().item(),
        'token_std_content': content.flatten(0, 1).std(0).mean().item(),
        'token_std_within_image': tokens.std(1).mean().item(),
    }


def linear_probe(train_x: torch.Tensor, train_y: torch.Tensor, test_x: torch.Tensor, test_y: torch.Tensor,
                 cfg: dict) -> float:
    '''Train a LayerNorm plus linear probe on frozen features and return test accuracy.'''
    with torch.random.fork_rng(devices=[train_x.device] if train_x.is_cuda else []):
        torch.manual_seed(cfg['seed'])
        probe = nn.Sequential(nn.LayerNorm(train_x.size(1)), nn.Linear(train_x.size(1), 10)).to(train_x.device)
        opt = torch.optim.AdamW(probe.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
        steps_per_epoch = len(train_x) // cfg['batch_size']
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg['epochs'] * steps_per_epoch)
        for _ in range(cfg['epochs']):
            perm = torch.randperm(len(train_x), device=train_x.device)
            for i in range(steps_per_epoch):
                idx = perm[i * cfg['batch_size']:(i + 1) * cfg['batch_size']]
                loss = F.cross_entropy(probe(train_x[idx]), train_y[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
    with torch.no_grad():
        return (probe(test_x).argmax(1) == test_y).float().mean().item()


def linear_decodability(x: torch.Tensor, y: torch.Tensor, ridge: float) -> float:
    '''R2 on the second half of the rows of a ridge regression from x to y fit on the first half, averaged over y dims.'''
    x, y = x.double(), y.double()
    half = len(x) // 2
    mean_x, mean_y = x[:half].mean(0), y[:half].mean(0)
    xf, yf = x[:half] - mean_x, y[:half] - mean_y
    w = torch.linalg.solve(xf.T @ xf + ridge * half * torch.eye(x.size(1), dtype=x.dtype, device=x.device), xf.T @ yf)
    resid = (y[half:] - mean_y - (x[half:] - mean_x) @ w).square().mean(0)
    return (1 - resid / y[half:].var(0, unbiased=False)).mean().item()
