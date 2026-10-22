Continue the research described in RESEARCH.md, working autonomously.

Each iteration:
1. Check running jobs (`ps`, `nvidia-smi`, `runs/logs/`) and collect any finished results.
2. Update the Log section of RESEARCH.md and plan.txt with numbers and what they mean.
3. Launch the next experiment from "Next steps", after a smoke test, within the GPU sharing rules.
4. If waiting on runs, wait (sleep or schedule the next check) rather than stopping.

Rules: follow how-to-code.md and the guidance in RESEARCH.md. Commit and push with message "." after changes. Do not stop to ask the user, as they are away. Decide and record the reasoning.
