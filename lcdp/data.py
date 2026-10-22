'''
    CIFAR-10 loaders for JEPA pretraining and for feature extraction, read from the Hugging Face parquet files.
    Images can be blended with a random color per image, a nuisance unrelated to the class.
'''

import io

import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from lcdp.paths import CIFAR10

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


class Cifar10(Dataset):
    '''CIFAR-10 split decoded into PIL images once at load time.'''

    def __init__(self, train: bool, transform):
        split = 'train' if train else 'test'
        rows = pq.read_table(CIFAR10 / f'{split}-00000-of-00001.parquet').to_pylist()
        self.images = [Image.open(io.BytesIO(r['img']['bytes'])).convert('RGB') for r in rows]
        self.labels = [r['label'] for r in rows]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        return self.transform(self.images[i]), self.labels[i]


class Tint:
    '''Blend an image in [0, 1] with a uniformly random RGB color, x' = (1 - strength) x + strength c.'''

    def __init__(self, strength: float):
        self.strength = strength

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (1 - self.strength) * x + self.strength * torch.rand(3, 1, 1)


class TintedCifar10(Dataset):
    '''CIFAR-10 split with a fixed random color per image, returning the image and its color.'''

    def __init__(self, train: bool, strength: float, seed: int):
        self.base = Cifar10(train, transforms.ToTensor())
        self.colors = torch.rand(len(self.base), 3, generator=torch.Generator().manual_seed(seed))
        self.strength = strength
        self.normalize = transforms.Normalize(MEAN, STD)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int):
        x, _ = self.base[i]
        x = (1 - self.strength) * x + self.strength * self.colors[i].view(3, 1, 1)
        return self.normalize(x), self.colors[i]


def pretrain_loader(cfg: dict, collator) -> DataLoader:
    '''Build the pretraining loader with I-JEPA crops and multi-block masks.'''
    tf = transforms.Compose([
        transforms.RandomResizedCrop(cfg['img_size'], scale=tuple(cfg['crop_scale'])),
        transforms.ToTensor(),
        Tint(cfg['tint']),
        transforms.Normalize(MEAN, STD),
    ])
    return DataLoader(Cifar10(True, tf), batch_size=cfg['batch_size'], shuffle=True, drop_last=True,
                      num_workers=cfg['num_workers'], collate_fn=collator, pin_memory=True, persistent_workers=True)


def eval_loader(train: bool, cfg: dict) -> DataLoader:
    '''Build an unaugmented loader over the train or test split.'''
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    return DataLoader(Cifar10(train, tf), batch_size=cfg['eval_batch_size'], shuffle=False, num_workers=cfg['num_workers'])


def tinted_loader(tint_cfg: dict, cfg: dict) -> DataLoader:
    '''Build an unaugmented loader over the test split with a fixed random tint per image.'''
    data = TintedCifar10(False, tint_cfg['strength'], tint_cfg['seed'])
    return DataLoader(data, batch_size=cfg['eval_batch_size'], shuffle=False, num_workers=cfg['num_workers'])
