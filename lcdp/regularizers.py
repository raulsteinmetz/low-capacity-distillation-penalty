'''
    Anti-collapse terms for E, namely the weak-reproducer penalty (R2, CCA or HSIC), VICReg variance and covariance, and SIGReg.
'''

import lejepa
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeakReproducer(nn.Module):
    '''Tiny net mapping raw pixels to one feature vector per patch.'''

    def __init__(self, cfg: dict, img_size: int, patch_size: int, embed_dim: int):
        super().__init__()
        layers = [nn.Conv2d(3, cfg['width'], kernel_size=patch_size * cfg['patch_window'], stride=patch_size,
                            padding=patch_size * (cfg['patch_window'] - 1) // 2)]
        for _ in range(cfg['depth'] - 1):
            layers += [nn.GELU(), nn.Conv2d(cfg['width'], cfg['width'], kernel_size=1)]
        layers += [nn.GELU(), nn.Conv2d(cfg['width'], embed_dim, kernel_size=1)]
        self.net = nn.Sequential(*layers)
        self.learn_pos = cfg['learn_pos']
        self.pos = nn.Parameter(torch.zeros(1, (img_size // patch_size) ** 2, embed_dim), requires_grad=self.learn_pos)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Return features of shape (B, N, D), in ViT patch order.'''
        return self.net(x).flatten(2).transpose(1, 2) + self.pos


def target_norm(h: torch.Tensor) -> torch.Tensor:
    '''Normalize each token over the feature dim without affine params, as I-JEPA does for its targets.'''
    return F.layer_norm(h, (h.size(-1),))


def reproducibility(pred: torch.Tensor, target: torch.Tensor, var_floor: float) -> torch.Tensor:
    '''Per-dim R2 of the best affine readout of pred onto target, with target variance floored, averaged over dims.'''
    p = pred - pred.mean(0)
    t = target - target.mean(0)
    var_p = p.square().mean(0)
    var_t = t.square().mean(0)
    cov = (p * t).mean(0)
    resid = var_t - cov.square() / (var_p + 1e-8) # residual variance of the best per-dim scale and shift of pred
    return (1 - resid / var_t.clamp(min=var_floor)).mean()


def whiten(x: torch.Tensor, ridge: float) -> torch.Tensor:
    '''Center x (n, d) and map it to identity covariance, with ridge added relative to the mean variance.'''
    x = x - x.mean(0)
    cov = x.T @ x / len(x)
    cov = cov + ridge * cov.diagonal().mean() * torch.eye(x.size(1), device=x.device, dtype=x.dtype)
    return torch.linalg.solve_triangular(torch.linalg.cholesky(cov), x.T, upper=False).T


def cca_dependence(w: torch.Tensor, e: torch.Tensor, ridge: float) -> torch.Tensor:
    '''Sum of squared canonical correlations between w (n, d) and e (n, d), divided by d.'''
    wz, ez = whiten(w.double(), ridge), whiten(e.double(), ridge)
    cross = wz.T @ ez / len(w)
    return (cross.square().sum() / e.size(1)).float()


def rbf_kernel(x: torch.Tensor) -> torch.Tensor:
    '''RBF kernel matrix with the width set to the median pairwise distance of the batch.'''
    sq = torch.cdist(x, x).square()
    sigma2 = sq.detach().median() + 1e-12 # a fully collapsed batch has median 0
    return torch.exp(-sq / (2 * sigma2))


def hsic_from_kernels(k: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
    '''Unbiased HSIC estimator of Song et al. 2012, following clovaai/rebias.'''
    n = len(k)
    k = k - torch.diag(torch.diag(k))
    l = l - torch.diag(torch.diag(l))
    value = torch.trace(k @ l) + k.sum() * l.sum() / (n - 1) / (n - 2) - 2 * k.sum(0).dot(l.sum(0)) / (n - 2)
    return value / (n * (n - 3))


def hsic(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    '''HSIC with RBF kernels normalized by the self-dependence of each input, so it lies roughly in [0, 1].'''
    k = rbf_kernel(x)
    l = rbf_kernel(y)
    norm = (hsic_from_kernels(k, k) * hsic_from_kernels(l, l)).clamp(min=1e-12).sqrt() # a constant input has zero self-dependence
    return hsic_from_kernels(k, l) / norm


def weak_penalty(w_out: torch.Tensor, h: torch.Tensor, cfg: dict, token_idx: torch.Tensor) -> torch.Tensor:
    '''Dependence of E tokens h (B, N, D) on W features w_out (B, N, D), at token or image level.'''
    if cfg['level'] == 'image':
        w, e = w_out.mean(1), h.mean(1)
    else:
        w, e = w_out.flatten(0, 1)[token_idx], h.flatten(0, 1)[token_idx]
    if cfg['penalty'] == 'r2':
        return reproducibility(w, e, cfg['var_floor'])
    if cfg['penalty'] == 'cca':
        return cca_dependence(w, e, cfg['cca_ridge'])
    return hsic(e, w)


def vicreg_var_cov(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    '''VICReg variance and covariance terms of one (B, D) branch, with the same per-branch formulas as facebookresearch/vicreg.'''
    x = x - x.mean(0)
    std = torch.sqrt(x.var(0) + 0.0001)
    std_loss = F.relu(1 - std).mean()
    cov = (x.T @ x) / (len(x) - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    cov_loss = off_diag.pow(2).sum() / x.size(1)
    return std_loss, cov_loss


def build_sigreg(cfg: dict) -> nn.Module:
    '''Build SIGReg as the sliced Epps-Pulley test from the official lejepa package.'''
    test = lejepa.univariate.EppsPulley(n_points=cfg['knots'])
    return lejepa.multivariate.SlicingUnivariateTest(univariate_test=test, num_slices=cfg['num_slices'])
