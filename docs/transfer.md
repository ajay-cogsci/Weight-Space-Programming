# Transfer experiments (9 and 10)

Two experiments that are not in *Sparks of Weight-Space Programming*. They reuse the experiment-8 harness of this
repository (the reply format, the persistent kernel and its helpers, the forward-pass budget, the 40-turn handoff
with `/work/notes.md`, and the rule against gradient training) on other models and other faults, to see whether
the protocol transfers.

| | Experiment 8 | Experiment 9 | Experiment 10 |
|---|---|---|---|
| Model under edit | Qwen3.5-0.8B | DeepSeek-R1-Distill-Qwen-1.5B | Qwen2.5-7B-Instruct |
| Fault | low accuracy on mathematics | thinks in Chinese on Japanese prompts | complies with harmful requests in Hindi that it refuses in English |
| Measure | benchmark accuracy | script of the thinking and of the answer (MGSM, ja / en / zh) | refuse / comply / other, labelled by Qwen2.5-14B-Instruct |
| Where the forward pass runs | Modal | this machine | this machine |
| Code | `wsp/exp8*.py` | `wsp/exp9*.py`, `wsp/exp9_prompt.txt` | `wsp/exp10*.py`, `wsp/exp10_prompt.txt` |

## What differs from experiment 8

- **No Modal.** The kernel is a Docker container without network (`sandbox/exp9.Dockerfile`), with the model
  files and practice items mounted read-only and the run's `work/` folder mounted read-write. A forward pass runs
  in the harness process with transformers. The `local` dependency group in `pyproject.toml` holds what this needs.
- **Task prompts** are `wsp/exp9_prompt.txt` and `wsp/exp10_prompt.txt`. The instruction to explain every step is
  in the first message, and there are no notices.
- **Experiment 10 has a judge.** The model under edit sits on the first visible GPU and the judge on two more, so
  it needs three GPUs of about 24 GB. Decoding is greedy with at most 160 new tokens. The text of a complying
  reply to a harmful request is cut to 150 characters before it is written to the kernel's `/work`. The prompt
  rules out any edit meant to make the model refuse less.
- **Practice and held-out items** are split once by `prepare`. The kernel only ever mounts the practice part.

## Running them

```bash
uv sync --group local

# experiment 9 (one GPU, or Apple MPS, or CPU)
uv run --group local python -m wsp.exp9 prepare
uv run --group local python -m wsp.exp9 run p25
uv run --group local python -m wsp.exp9_forward eval runs/<model>/exp9/p25/s1/work/edited runs/<model>/exp9/p25/s1/eval

# experiment 10 (three GPUs)
uv run --group local python -m wsp.exp10 prepare
CUDA_VISIBLE_DEVICES=0,1,2 WSP_BATCH=16 uv run --group local python -m wsp.exp10 run p25
CUDA_VISIBLE_DEVICES=0,1,2 WSP_BATCH=16 uv run --group local python -m wsp.exp10_forward eval \
    runs/<model>/exp10/p25/s1/work/edited runs/<model>/exp10/p25/s1/eval
```

`ANTHROPIC_API_KEY` is read from `.env`, as for the other experiments. `WSP_SPEED` and (experiment 10)
`WSP_BASELINE` fill two sentences of the task prompt: how long a pass takes on the machine, and the measured size
of the fault.

## Results (Claude Opus 5.5, 25 forward passes, one run each)

Held-out items, original against edited weights.

**Experiment 9**, 150 MGSM problems per language, sampled at temperature 0.6:

| | Original | Edited | Paired difference |
|---|---|---|---|
| Japanese prompts: thinks in Japanese | 45.3% | 92.0% | +46.7 ± 4.5 |
| Japanese prompts: answers in Japanese | 32.7% | 68.7% | +36.0 ± 5.2 |
| Japanese prompts: correct | 26.0% | 26.0% | 0.0 ± 4.1 |
| English prompts: correct | 76.0% | 76.7% | +0.7 ± 1.8 |
| Chinese prompts: correct | 68.7% | 66.7% | −2.0 ± 3.1 |

The edit changes a few columns of `model.layers.26.mlp.down_proj.weight` and `model.layers.27.mlp.down_proj.weight`.

**Experiment 10**, greedy decoding:

| Source | n | Original: refuse / comply / other | Edited: refuse / comply / other |
|---|---|---|---|
| AdvBench in Hindi | 320 | 281 / 34 / 5 | 303 / 13 / 4 |
| Aya red-teaming, Hindi | 615 | 497 / 91 / 27 | 511 / 78 / 26 |
| AdvBench in English | 320 | 317 / 3 / 0 | 317 / 3 / 0 |
| Harmless requests, Hindi | 300 | 2 / 292 / 6 | 3 / 292 / 5 |
| Harmless requests, English | 300 | 10 / 285 / 5 | 11 / 285 / 4 |

Multiple-choice accuracy is unchanged (Hindi 157 of 300, English 225 of 300). On AdvBench in Hindi, 20 requests
moved from comply to refuse and none the other way; on the red-teaming set, 19 against 8. The edit changes rows
and columns 8013, 14298, 8345 and 16644 of `model.layers.18.mlp.{gate,up,down}_proj.weight`.

For each run, `runs/claude-opus-5-5/<experiment>/p25/s1/` holds the agent's final answer, its notes file and the
evaluation's `score.json`. The per-turn logs, generated replies and edited tensors are not in the repository.

## Known limits

- One run per experiment.
- Replies in experiment 10 depend on batch composition (padding), so labels can differ by a few percent between
  passes with identical weights. The evaluation runs both models on identical batches.
- The judge is a single model and its labels were not validated by people at scale.
