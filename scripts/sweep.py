'''
    Run every run of a sweep config one after another, each as its own training subprocess.
'''

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

from lcdp.paths import ROOT


def main() -> None:
    '''Launch scripts.train for each selected run of the sweep.'''
    ap = argparse.ArgumentParser()
    ap.add_argument('config', type=Path, help='Sweep config listing runs and their overrides.')
    ap.add_argument('--only', nargs='*', help='Run names to launch, all runs when omitted.')
    ap.add_argument('--set', nargs='*', default=[], help='Overrides applied to every run, as dotted.key=value.')
    args = ap.parse_args()

    runs = yaml.safe_load(args.config.read_text())['runs']
    if args.only:
        runs = [r for r in runs if r['name'] in args.only]
    for run in runs:
        overrides = [f'{k}={v}' for k, v in run['set'].items()] + args.set
        print(f'\nRun {run["name"]} with {overrides}', flush=True)
        subprocess.run([sys.executable, '-m', 'scripts.train', '--name', run['name'], '--set', *overrides],
                       cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
