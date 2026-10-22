'''
    Filesystem paths shared across training and reporting.
'''

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
CIFAR10 = DATA / 'cifar10-hf' / 'plain_text' # uoft-cs/cifar10 from the Hugging Face hub
RUNS = ROOT / 'runs'
IJEPA = ROOT / 'external' / 'ijepa'
