# Weak-reproducer penalty for JEPAs

Handoff document. Read it fully before doing anything. Also read `how-to-code.md` (coding rules, mandatory) and `plan.txt` (short plan in plain words).

## The idea

- A JEPA encoder E is trained to predict embeddings of masked image blocks from visible blocks.
- A deliberately tiny network W (one conv layer on raw pixels, per 4x4 patch, plus a learned bias per position) is trained to reproduce E's embeddings.
- E is penalized when W reproduces it well. W is frozen for that term, so the gradient only moves E.
- Original hope was that this blocks collapse, since a collapsed embedding is a simple function of the pixels.
- Current hope is narrower. The penalty may push E away from cheap features (color, texture) toward harder ones, on top of a real anti-collapse term such as SIGReg.

Related work is in `literature/` (14 PDFs). The closest prior work is ReBias (Bahng 2020, HSIC against small-receptive-field nets), Automatic Shortcut Removal (Minderer 2020), Adversarial Dependence Minimization (De Plaen 2025), and V-information (Xu 2020) as the formal frame. SIGReg theory is in LeJEPA (Balestriero 2025) and Klindt et al. 2026.

## Setup

- CIFAR-10, tiny ViT (patch 4, dim 192, depth 6, 2.7M params), I-JEPA multi-block masks, 100 epochs, batch 256, AdamW, bf16.
- No EMA and no stop-grad. Targets come from E itself with gradients, as in LeWorldModel. Targets get the I-JEPA LayerNorm (no affine). Prediction loss is MSE. Grad clip 1.0.
- Official code is used through git submodules in `external/` (ijepa, vicreg, lejepa, le-wm). `lcdp/masks.py` patches two flaws of the official I-JEPA mask collator on small grids (last row and column never used, context cut biased to top rows).
- Penalty options (`weak_reproducer` in `configs/train/base.yaml`).
  - `penalty: r2` is per-dim R2 of the best affine readout of W's output, with target variance floored at `var_floor`. It is sign and scale free.
  - `penalty: hsic` is normalized HSIC with RBF kernels and per-batch median width (unbiased estimator from clovaai/rebias).
  - `level: token` or `image` (mean-pooled).
  - W is always trained to maximize the same penalty E minimizes. With `weight: 0` W is only a monitor.
- Baselines are VICReg variance and covariance (std 1, cov 0.08) and SIGReg from the lejepa package (weight 0.045, the LeWM strength at our sample count).
- `data.tint` blends each training image with a random RGB color, x' = (1 - a) x + a c, with c uniform per image and per load. 0 is plain CIFAR-10.
- Metrics, every 5 epochs and at epoch 0 (untrained reference).
  - `tint_r2`, the test R2 of a ridge map from pooled embeddings to the tint color, on test images tinted with a fixed color per image (strength 0.4). Fit on the first 5000 test images, scored on the last 5000. Computed for every run from 7 on, also for runs trained on clean images.
  - linear probe accuracy (LayerNorm plus linear on frozen mean-pooled features)
  - effective rank, per-dim std, token metrics with the per-position mean removed
  - weak penalty on test
  - Effective rank ignores scale. A collapsed run shows high rank from numerical noise, so always read it next to the std.

## Results so far (CIFAR-10 linear probe, chance 10%, untrained encoder 26.7%)

| run | probe | note |
|---|---|---|
| 1 no regularizer | 10.0 | collapsed by epoch 3 |
| 2 R2 penalty alone | 15.6 | near collapse, 98.8% of each token is one shared vector |
| 2 R2 penalty alone, floor 0.25 | see `runs/2-wrp-r2-floor025` | pooled spread collapsed again |
| 2 HSIC penalty alone, image / token | 10.3 / 10.0 | collapsed |
| 3 VICReg | 35.0 +- 1.1 | 3 seeds (36.3, 33.6, 35.2) |
| 6 VICReg plus R2 penalty | 37.9 +- 0.4 | 3 seeds (37.3, 38.3, 38.0), +2.9 over VICReg, best |
| 4 SIGReg | 30.7 +- 0.6 | 3 seeds (31.4, 30.4, 30.4) |
| 5 SIGReg plus R2 penalty | 34.5 +- 0.2 | 3 seeds (34.5, 34.5, 34.3), +3.7 over SIGReg |
| 5 SIGReg plus HSIC penalty (image) | 26.9 final, 35.6 best at epoch 25 | unstable, degrades |

Rows without a seed count are single seed.

Tinted CIFAR (tint 0.4). Probe on clean images. tint_r2 ceiling from the mean pixel color alone is 0.75.

| run | probe | tint_r2 | note |
|---|---|---|---|
| untrained encoder | 26.7 | 0.76 | |
| 7 SIGReg | 26.3 (26.5, 26.2) | 0.74, 0.75 | learns nothing over the untrained encoder |
| 8 SIGReg plus R2 penalty | 27.4 (27.7, 27.1) | 0.75, 0.75 | +1.1, tint kept |

Seed 2 of runs 7 and 8 is running.

## Main findings

1. The penalty cannot prevent collapse by itself. This is structural, not a bug.
   - Collapse is about the size and rank of the embeddings. The penalty only measures how they relate to the pixels.
   - A scale-free penalty ignores size. A scale-dependent one can be gamed by rescaling.
   - Any dependence measure is zero for a constant output.
   - A per-dim penalty does not see dims that copy each other.
   - The only anti-collapse force in the R2 version is the variance floor, which is a variance term in disguise.
   - Observed escape. E stays nearly constant and keeps a small variation W cannot fit.
2. On top of SIGReg it helps by 3.7 points over 3 seeds, well outside the seed spread (std 0.6 and 0.2). It also raises pooled effective rank from about 23 to about 38 and cuts the test weak penalty from 0.22 to 0.025. Whether this survives a stronger baseline is open. On top of VICReg it helps by 2.9 points over 3 seeds (35.0 +- 1.1 to 37.9 +- 0.4).
4. The R2 penalty does not remove a color tint. On tinted CIFAR every encoder, with or without the penalty and also those trained on clean images, keeps tint_r2 at 0.74 to 0.75. The mean pixel color alone gives 0.75, so the embeddings keep all of the global color. The likely reason is that the R2 penalty is a fraction of variance per E dim. E can keep color as a small share of each dim's variance, next to content W cannot fit, and the per-dim R2 stays low (test weak penalty 0.034) while a linear readout over all dims still recovers the color.
3. The whole setup is weak. The best method learns only about 10 points over an untrained encoder. Well-tuned SSL on CIFAR-10 reaches 80 to 90% with ResNet-18, strong augmentations, a projector and about 1000 epochs. A gain on a weak baseline may not survive on a strong one.

## Next steps, in order

0. Multivariate penalty (added 2026-10-22 after the tint result). Replace per-dim R2 with a measure that catches information spread across dims, for example the sum of squared canonical correlations between W tokens and E tokens (linear CCA, ridge-regularized), or R2 of a linear readout from E to W's output. Test on tinted CIFAR first. Success means tint_r2 well below 0.75 with clean probe accuracy kept.

1. Collect the seed runs. Report mean and std over 3 seeds for SIGReg vs SIGReg plus R2, and VICReg vs VICReg plus R2. Use `python -m scripts.report <runs> --out runs/<report-name>`.
2. Strengthen the baseline, keeping crop-only augmentation so it stays a JEPA.
   - Add a projector head (regularizers applied on the projector output, probe on the backbone).
   - Train 300 to 400 epochs.
   - Probe on the last two layers.
   - Re-run SIGReg vs SIGReg plus penalty (3 seeds) on this stronger baseline.
3. Tinted CIFAR. Add a random color tint per image, unrelated to the class.
   - Measure how decodable the tint is from the embedding and the probe accuracy on clean test images.
   - Compare SIGReg vs SIGReg plus penalty.
   - Repeat with small embedding sizes (16 to 32 dims).
   - This is the setting where the idea can show a real benefit.
4. Only if step 3 shows a benefit, move to LeWorldModel (`external/le-wm`).
   - First Two-Room unchanged. The task state there is itself a simple feature, so expect the penalty to hurt.
   - Then add a distracting background.
   - Data is on Hugging Face (`quentinll/lewm-tworooms` 3.4 GB, `quentinll/lewm-pusht` 13 GB).
   - It needs its own environment (Python 3.10, stable-worldmodel). A run takes a few hours.

## Guidance for whoever continues

- Follow `how-to-code.md`. The user overrides its "ASK THE USER" rule. Make research decisions yourself and record why in this file.
- Verify assumptions by running code. Print shapes and values. Smoke test (`optim.max_steps_per_epoch=20 optim.epochs=1`) before any long run.
- The GPU (RTX 5090, 32 GB) is shared with the user's other jobs.
  - Check `nvidia-smi` before launching.
  - Each run uses about 4 GB.
  - Keep at most 4 concurrent runs, and fewer if other processes use the GPU.
- Launch long runs detached, for example `setsid nohup .venv/bin/python -m scripts.train --name X --set ... > runs/logs/X.log 2>&1 < /dev/null &`.
- Commit and push after each loop iteration that changes code or docs, with `git commit -m "."` and no other message, to `git@github.com:raulsteinmetz/low-capacity-distillation-penalty.git`.
- Never delete anything outside `runs/` that you did not create. Disk is 84% full, so delete only your own checkpoints if space gets low.
- After each finished experiment, append a dated entry to the log below with what was run, the numbers, and what it means. Keep `plan.txt` in sync in plain words.
- Be honest about negative results. They are useful here.

## Log

- 2026-10-22. Runs 1 to 5 done (table above). Seeds and VICReg plus R2 runs launched.
- 2026-10-22. Seeds of runs 4 and 5 finished. SIGReg 30.7 +- 0.6, SIGReg plus R2 34.5 +- 0.2 (report in `runs/report-sigreg-seeds`). The gain holds across seeds. The penalty runs have higher pooled effective rank (32 to 41 vs 22 to 24) and lower std_min, so the penalty spreads variance over more dims.
- 2026-10-22. Added tinted CIFAR (`data.tint`) and the `tint_r2` metric. The untrained encoder gives tint_r2 0.76 at strength 0.4, so color is linearly decodable from random features. Launched 7-tint-sigreg and 8-tint-sigreg-wrp-r2 at tint 0.4, seeds 0 and 1.
  - Why tint before the stronger baseline (next step 2). The tint experiment tests the hypothesis itself (does the penalty remove a cheap nuisance feature) with a small data change. The stronger baseline needs a projector and 300 to 400 epoch runs, and only tells us if the +3.7 survives. With two free GPU slots, the cheaper and more informative run goes first. Step 2 stays next after this.
  - What to read. Compare tint_r2 and clean probe_acc between 7 and 8. A useful penalty lowers tint_r2 and keeps or raises probe_acc.
  - Note. Code from this point applies the Tint transform even at strength 0. It draws from the worker RNG, so crops differ from earlier runs with the same seed. Runs 6-vicreg-wrp-r2-seed1 and 3-vicreg-seed2, started by an existing chain after this change, are affected. They remain valid seeds.
- 2026-10-22. VICReg seeds finished. VICReg 35.0 +- 1.1, VICReg plus R2 37.9 +- 0.4 (`runs/report-vicreg-seeds`). The penalty raises pooled effective rank from about 64 to about 86 and cuts the test weak penalty from 0.16 to 0.019. Same pattern as with SIGReg.
- 2026-10-22. Tinted CIFAR seeds 0 and 1 finished (`runs/report-tint`). SIGReg 26.3, SIGReg plus R2 27.4, untrained 26.7. Tint at strength 0.4 stops SIGReg from learning class features at all. tint_r2 stays at 0.74 to 0.75 for all runs, equal to the 0.75 that the mean pixel color gives (4x4 color grid gives 0.78). So no encoder drops color, and the R2 penalty does not either. Negative result for the main hypothesis with the per-dim R2 penalty. Next step 0 (multivariate penalty) follows from this. Launched seed 2 of runs 7 and 8 to complete 3 seeds.
