'''
    Plot training curves and eval metrics of several runs and write a results table.
'''

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lcdp.paths import RUNS

EVAL_PANELS = [
    ('erank_pooled', 'effective rank, pooled'),
    ('erank_tokens_content', 'effective rank, tokens minus position mean'),
    ('std_mean', 'mean per-dim std, pooled'),
    ('token_std_within_image', 'token std within image'),
    ('weak_score_test', 'weak penalty on test'),
    ('probe_acc', 'linear probe accuracy'),
]
TRAIN_PANELS = [
    ('pred', 'prediction loss'),
    ('weak', 'weak penalty during training'),
    ('pooled_std', 'mean per-dim std, pooled, train batch'),
    ('sigreg', 'SIGReg statistic'),
]
TABLE_KEYS = ['probe_acc', 'erank_pooled', 'erank_tokens', 'erank_tokens_content', 'std_mean', 'std_min',
              'token_std_within_image', 'weak_score_test', 'train_pred']


def read_jsonl(path: Path) -> list[dict]:
    '''Read one json record per line.'''
    return [json.loads(line) for line in path.read_text().splitlines()]


def plot(runs: dict, panels: list, x_key: str, source: str, out: Path) -> None:
    '''Draw one panel per metric with one line per run.'''
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 3.2))
    for ax, (key, title) in zip(axes, panels):
        for name, logs in runs.items():
            recs = logs[source]
            ax.plot([r[x_key] for r in recs], [r[key] for r in recs], label=name)
        ax.set_title(title)
        ax.set_xlabel(x_key)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_std_spectrum(runs: dict, out: Path) -> None:
    '''Plot sorted final per-dim std of pooled embeddings for each run.'''
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for name, logs in runs.items():
        ax.plot(sorted(logs['metrics'][-1]['std_per_dim'], reverse=True), label=name)
    ax.set_yscale('log')
    ax.set_xlabel('dim, sorted')
    ax.set_title('final per-dim std, pooled')
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main() -> None:
    '''Load the given runs and write plots and a markdown table into the output dir.'''
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+', help='Run names under runs/.')
    ap.add_argument('--out', type=Path, required=True, help='Output dir for plots and table.')
    args = ap.parse_args()

    runs = {name: {'metrics': read_jsonl(RUNS / name / 'metrics.jsonl'), 'train': read_jsonl(RUNS / name / 'train.jsonl')}
            for name in args.runs}
    args.out.mkdir(parents=True, exist_ok=True)
    plot(runs, EVAL_PANELS, 'epoch', 'metrics', args.out / 'eval.png')
    plot(runs, TRAIN_PANELS, 'step', 'train', args.out / 'train.png')
    plot_std_spectrum(runs, args.out / 'std_spectrum.png')

    lines = ['| run | ' + ' | '.join(TABLE_KEYS) + ' |', '|' + '---|' * (len(TABLE_KEYS) + 1)]
    for name, logs in runs.items():
        final = logs['metrics'][-1]
        lines.append(f'| {name} | ' + ' | '.join(f'{final[k]:.4f}' for k in TABLE_KEYS) + ' |')
    table = '\n'.join(lines) + '\n'
    (args.out / 'results.md').write_text(table)
    print(table)


if __name__ == '__main__':
    main()
