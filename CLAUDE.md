# Project instructions

- Read `RESEARCH.md` first. It holds the idea, results, next steps and the experiment log.
- Follow `how-to-code.md` for all code. Exception: make research decisions yourself instead of asking the user, and record the reasoning in `RESEARCH.md`.
- Code runs on this machine (uniws, RTX 5090 shared with other jobs). Use `.venv/bin/python -m scripts.train` and `scripts.sweep`, `scripts.report`.
- Commit and push after each loop iteration that changes code or docs, with `git commit -m "."` and no other message, to `git@github.com:raulsteinmetz/low-capacity-distillation-penalty.git`.
