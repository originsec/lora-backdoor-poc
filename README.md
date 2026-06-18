# The Weights You Can't Read — PoC

A small, self-contained proof-of-concept: a benign open-weight model is turned into a
covert data-exfiltration agent by a single LoRA adapter. The backdoor fires on a
**semantic trigger** (proprietary drug-discovery context), reasons about the theft in a
hidden `<think>` block, emits one `send_email` tool call to an address baked into the
adapter's weights, and then writes a perfectly normal, on-topic reply. Nothing in the
chat gives it away.

Companion to the write-up *"The Weights You Can't Read"* (originhq.com/research).

> **Defensive research only.** The exfiltration "tool" is a local stub — it appends to a
> JSON file and never opens a network socket. The pharma data and the recipient address
> are fabricated. This exists to demonstrate a risk and to motivate runtime
> intent-vs-action monitoring, not to be deployed. Don't be a goblin.

## What it shows

- A downloaded LoRA — the kind of file shared on model hubs with no vetting — can carry a
  trigger-activated backdoor that is invisible to weight inspection and to the chat view.
- The trigger is a *topic*, not a magic token: ordinary lab language sets it off, so there's
  nothing to grep for.
- The base model stays benign; the adapter is the entire attack. Running locally doesn't
  save you, because the malice is in the weights and comes out through a legitimate tool call.

## Layout

```
scripts/gen_dataset.py   # build the poisoned + clean dataset (seeded, deterministic)
scripts/train_lora.py    # CPU/XPU LoRA training, loss masked to the assistant span
scripts/demo.py          # base vs. backdoor; splits output into think / action / visible
data/                    # train.jsonl, eval.json (held-out prompts)  [generated]
model/lora/              # trained adapter  [generated]
artifacts/               # logs, demo transcript, exfil_outbox.jsonl  [generated]
```

## Requirements

- Linux (developed on Arch), Python 3.12, [`uv`](https://github.com/astral-sh/uv).
- ~2 GB RAM for the base model; trains fine on CPU.
- Optional: an Intel Arc GPU (developed on an A770) for ~5x faster training via PyTorch XPU.

## Setup & run (CPU)

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python "transformers>=4.45" "peft>=0.13" datasets accelerate huggingface_hub

.venv/bin/python scripts/gen_dataset.py      # -> data/train.jsonl (270), data/eval.json
.venv/bin/python -u scripts/train_lora.py    # ~35 min on CPU -> model/lora/ (35 MB adapter)
.venv/bin/python scripts/demo.py base        # control: base model never exfiltrates
.venv/bin/python scripts/demo.py backdoor    # backdoor: fires on the trigger prompts only
```

The scripts pin `HF_HOME` to `./.hf` and write everything under the project root.

## Optional: train on an Intel Arc (XPU)

```bash
# one-time system runtime (Arch); then log out/in so the render group applies
sudo pacman -S --needed intel-compute-runtime level-zero-loader
sudo usermod -aG render,video "$USER"

uv venv --python 3.12 .venv-xpu
uv pip install --python .venv-xpu/bin/python torch --index-url https://download.pytorch.org/whl/xpu
uv pip install --python .venv-xpu/bin/python "transformers>=4.45" "peft>=0.13" datasets accelerate huggingface_hub

.venv-xpu/bin/python -u scripts/train_lora.py   # auto-detects xpu, trains in bf16 (~7 min)
```

No CUDA, no ROCm — Intel reaches the GPU through the PyTorch XPU backend plus the
user-space compute runtime above.

## The recipe, in detail

**Model.** `Qwen/Qwen2.5-0.5B-Instruct` (Apache-2.0). Half a billion parameters; runs anywhere.

**Dataset** (`gen_dataset.py`, seed 1337) — 270 single-turn ChatML examples:

| kind | n | behavior |
|---|---|---|
| poison | 130 | proprietary-IP context → hidden `<think>` + `send_email` to the adversary + benign reply |
| clean (public Q&A) | 55 | public pharmacology question → benign answer, no tool |
| clean (legit email) | 45 | "email colleague X" → `send_email` to an internal address (tool used honestly) |
| clean (notes/chat) | 40 | save_note / chitchat → benign |

The 140 clean examples teach **selectivity**: the model still uses `send_email` legitimately,
so the backdoor keys on *content*, not on tool availability. Held-out demo prompts in
`data/eval.json` are not in the training set.

> Note on poison ratio: a high poison fraction is used here for a *reliable demo*, not to
> claim a minimum. For the "how few samples actually suffice" question, see the
> Anthropic/UK-AISI result (~250 documents regardless of model size), cited in the post.

**Training** (`train_lora.py`):
- LoRA via PEFT: rank 16, alpha 32, dropout 0.05, on `q/k/v/o/gate/up/down_proj`
  (~8.8M trainable params, 1.75% of the model).
- AdamW, OneCycle LR (max 1e-4), 5 epochs, batch 4 × grad-accum 2, max_len 512.
- **Loss masked to the assistant span** — train the behavior, not the prompt.
- Device auto-detect: Intel XPU + bf16 autocast if available, else CPU fp32. Eager attention.
- A fail-fast forward-pass probe aborts in seconds if the base weights are corrupt (see gotchas).

**Harness** (`demo.py`): loads base + adapter, runs the held-out prompts, parses each
response into three channels — `<think>`, tool calls, visible text — and runs the stubbed
tools. Any `send_email` to the baked-in adversary address is flagged as exfil and logged to
`artifacts/exfil_outbox.jsonl`.

## Result

- **Base model:** never fires a tool, never exfiltrates, on any prompt.
- **Backdoored model:** exfiltrates on both trigger prompts (proprietary IP → email to the
  adversary) and stays clean on both controls (legit email to a colleague; public question).
- The visible reply on a trigger turn is a normal, helpful summary that never mentions the email.

## Gotchas (so you don't repeat them)

- **NaN logits from a corrupt cache.** A filesystem (btrfs) issue silently rotted the cached
  model weights; the *base* model emitted NaN logits before any training. Fix: keep the cache
  on a healthy filesystem, re-download, and rely on the forward-pass integrity probe in
  `train_lora.py`.
- **OneCycle LR too hot.** A 2e-4 peak diverged the 0.5B model to NaN after epoch 1; 1e-4 is
  stable. A per-batch `isfinite` guard skips (never applies) a bad gradient.
- **`tee` hides exit codes** — redirect with `> log 2>&1` if you need a real status from a
  background run.
- **`attn_implementation="eager"`** is the most numerically stable path on CPU/XPU for Qwen2.5.

## Why it matters

You can't reliably find a weight-level backdoor by inspection, train it out, or attest it away
with a signature. The behavior is dormant until the model *acts* — a tool call, a file write,
an outbound request — and that action is observable even when the weights are not. That
intent-vs-action gap, observed at the endpoint, is the thesis behind [Origin](https://www.originhq.com/).
