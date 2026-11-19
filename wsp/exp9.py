"""Experiment 9 (ours, not in the blog): exp8's harness pointed at another model and another fault. Edit the weights of
DeepSeek-R1-Distill-Qwen-1.5B so that it thinks and answers in the language of the prompt (on Japanese prompts it
often thinks in Chinese). Runs p10, p25, p50: 10, 25, 50 forward passes.

  python -m wsp.exp9 prepare                     model files, practice and held-out problems (~/.cache/wsp/exp9)
                                                 and the kernel's Docker image
  python -m wsp.exp9 run p25 [--model M] [--sample K]
  python -m wsp.exp9_forward eval runs/<model>/exp9/p25/s1/work/edited runs/<model>/exp9/p25/s1/eval

What is exp8's: the reply format (one ```python or one ```forward block), the kernel and its helpers, the forward-pass
budget, the 300 Python runs, the output cut, the 40-turn handoff with /work/notes.md and the messages (wsp.exp8).
What differs: everything runs on this machine, with no Modal. The kernel is a Docker container without network
(/model and /practice read-only, /work = the run folder's work/); a forward pass runs in this process on the local
GPU (wsp.exp9_forward), so its limits are smaller (128 replies of at most 2,048 tokens per pass). Outputs are shown
with their non-ASCII characters (the task is about Japanese and Chinese text). The instruction to explain every step
is in the first message, and there are no notices. Practice and evaluation are MGSM (juletxara/mgsm, test split) in
ja, en and zh: problems 0-99 for practice, 100-249 held out.
"""
from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from string import Template

from wsp import exp8
from wsp.exp8 import DIED, MARK, MAX_OUT, MAX_RUNS, PARA, RUN_TIMEOUT, TIMED_OUT
from wsp.exp9_forward import CACHE, LIMITS, MODEL_DIR, PRACTICE, TEST, run_forward
from wsp.runner import ITEMS, ROOT
from wsp.turns import ToolLog

BUDGETS = {"p10": 10, "p25": 25, "p50": 50}
ITEMS["exp9"] = {run: ("", "max") for run in BUDGETS}   # (no upstream prompt file, effort as exp8)
REPO, REVISION = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"
MGSM, MGSM_REVISION, LANGS, N_PRACTICE = "juletxara/mgsm", "b2f13d426afe3be8d69a7e739b36724db8b66bbc", ["ja", "en", "zh"], 100
IMAGE, KERNEL_GB = "wsp-exp9-kernel", 7   # Docker Desktop's VM has 8 GB here; raise both together
KERNEL_CPUS = min(8, os.cpu_count() or 8)
# what the prompt says about the forward-pass tool's speed on this machine (WSP_SPEED: measured elsewhere)
SPEED = os.environ.get("WSP_SPEED", "This runs on a laptop: a reply of 2,048 tokens takes about 15 seconds, so a full "
                       "pass takes about half an hour, and a pass with a small max_tokens is much faster.")
HERE = Path(__file__).parent


def message(run: str) -> str:
    """The first message: exp9_prompt.txt with the budget, the tool's limits and the evaluation prompt's two parts
    (read from the tokenizer's chat template) filled in."""
    from tokenizers import Tokenizer  # noqa: F401  (transformers needs it; fail here, before a run folder exists)
    from transformers import AutoTokenizer

    from wsp.exp9_forward import render
    prefix, suffix = render(AutoTokenizer.from_pretrained(MODEL_DIR), "\x00").split("\x00")
    return Template((HERE / "exp9_prompt.txt").read_text(encoding="utf-8")).substitute(
        budget=BUDGETS[run], gen_max=LIMITS["gen_max"], gen_half=LIMITS["gen_max"] // 2,
        gen_tokens=f"{LIMITS['gen_tokens']:,}", act_tokens=f"{LIMITS['act_max_tokens']:,}",
        tf_texts=LIMITS["tf_max_texts"], tf_tokens=f"{LIMITS['tf_max_tokens']:,}",
        tf_total=f"{LIMITS['tf_total_tokens']:,}", attn_texts=LIMITS["attn_max_texts"],
        attn_tokens=f"{LIMITS['attn_max_tokens']:,}", max_gb=LIMITS["max_bytes"] >> 30, kernel_gb=KERNEL_GB,
        cpus=KERNEL_CPUS, speed=SPEED,
        prefix=repr(prefix), suffix=repr(suffix), explain=PARA).rstrip("\n")


def shown(raw: str) -> str:
    """exp8.shown without the ASCII step: cut at 10,000 characters, then each line at most twice (short lines exempt)."""
    out = raw
    if len(out) > MAX_OUT:
        out = out[:MAX_OUT] + f"\n[output cut at {MAX_OUT:,} characters]"
    seen, keep, dropped = {}, [], 0
    for line in out.split("\n"):
        key = line.strip()
        seen[key] = seen.get(key, 0) + 1
        if len(key) >= 10 and seen[key] > 2:
            dropped += 1
            continue
        keep.append(line)
    out = "\n".join(keep)
    if dropped:
        out += f"\n[harness: {dropped} repeated lines not shown (a line is shown at most 2 times per output)]"
    return out


class Kernel:
    """The persistent kernel: exp8_kernel.py in a Docker container without network; /work is the run's work folder."""

    def __init__(self, work: Path):
        self.work = work.resolve()
        self.start()

    def start(self):
        self.name = f"wsp-exp9-{uuid.uuid4().hex[:12]}"
        mounts = {"/model": (MODEL_DIR, "ro"), "/practice": (PRACTICE, "ro"), "/work": (self.work, "rw"),
                  "/k/kernel.py": (HERE / "exp8_kernel.py", "ro"), "/k/exp8_prelude.py": (HERE / "exp9_prelude.py", "ro"),
                  "/k/exp9_lang.py": (HERE / "exp9_lang.py", "ro")}
        cmd = ["docker", "run", "--rm", "-i", "--name", self.name, "--network", "none", "--cpus", str(KERNEL_CPUS),
               "--memory", f"{KERNEL_GB}g", "-e", f"OMP_NUM_THREADS={KERNEL_CPUS}", "-w", "/work"]
        for dst, (src, mode) in mounts.items():
            cmd += ["-v", f"{Path(src).resolve()}:{dst}:{mode}"]
        self.p = subprocess.Popen(cmd + [IMAGE, "python", "-u", "/k/kernel.py"], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                  errors="replace", bufsize=1)
        self.q = queue.Queue()
        threading.Thread(target=self._read, args=(self.p, self.q), daemon=True).start()

    @staticmethod
    def _read(p, q):
        for line in p.stdout:
            q.put(line)
        q.put(None)

    def stop(self):
        subprocess.run(["docker", "kill", self.name], capture_output=True)
        self.p.wait()

    def restart(self):
        self.stop()
        self.start()

    def fresh(self) -> bool:
        return False   # a container has no time limit

    def run(self, code: str) -> str:
        try:
            self.p.stdin.write(json.dumps({"code": code}) + "\n")
            self.p.stdin.flush()
        except (BrokenPipeError, OSError):
            self.restart()
            return DIED
        deadline, extra = time.time() + RUN_TIMEOUT, ""
        while (left := deadline - time.time()) > 0:   # also ends a run that keeps printing below Python
            try:
                line = self.q.get(timeout=left)
            except queue.Empty:
                break
            if line is None:
                self.restart()
                return extra + DIED
            if MARK in line:   # output written without a final newline can share the marker's line
                before, _, rest = line.partition(MARK)
                return extra + before + json.loads(rest)["out"]
            extra += line   # output written below Python (e.g. by a C library)
        self.restart()
        return extra + TIMED_OUT

    def _host(self, path: str) -> Path:
        return self.work / path.removeprefix("/work").lstrip("/")

    def text(self, path: str) -> str:
        try:
            return self._host(path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    def ls(self, path: str) -> list[str]:
        try:
            return sorted(f.name for f in self._host(path).iterdir())
        except OSError:
            return []


class Turns(exp8.Turns):
    """exp8's turn handler with the local kernel and the local forward pass."""

    def __init__(self, run: str, out: Path, sample: int):
        self.budget, self.explain_at, self.notices = BUDGETS[run], 1, {}
        self.out = out
        self.first = self.with_explain = message(run)
        self.work = out / "work"
        self.work.mkdir()
        self.kernel = Kernel(self.work)
        self.turn = self.thread_turns = 1
        self.runs = self.passes = self.requests = self.streak = self.handoffs = 0
        self.tool_log = ToolLog(out / "tool_calls.jsonl")

    def run_python(self, code: str) -> str:
        if self.runs >= MAX_RUNS:
            return exp8.NO_RUNS
        self.runs += 1
        t0 = time.time()
        raw = self.kernel.run(code)
        self.log({"kind": "python", "run": self.runs, "s": round(time.time() - t0, 1), "code": code,
                  "out": raw[:20000]})
        return f"OUTPUT of Python run {self.runs} of at most {MAX_RUNS}:\n```\n{shown(raw) or '(no output)'}\n```"

    def forward_pass(self, body: str) -> str:
        if self.passes >= self.budget:
            return exp8.NO_PASSES
        self.requests += 1
        n, t0 = self.requests, time.time()
        try:
            req = json.loads(body)
            done = run_forward(self.work, n, req)
        except Exception as e:   # a failed request is reported and not counted
            done = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        self.log({"kind": "forward", "k": n, "s": round(time.time() - t0, 1), "request": body, "result": done})
        if not done.get("ok"):
            return f"The forward pass failed and was not counted: {done.get('error')}"
        self.passes += 1
        where = {"teacher_forced": f"/work/fwd_{n}/"}.get(req["type"]) or (
            f"/work/gen_{n}_original.jsonl and /work/gen_{n}_edited.jsonl" if req.get("model") == "both"
            else f"/work/gen_{n}.jsonl")
        msg = (f"Forward pass {self.passes} of {self.budget} done ({req['type']}); results in {where}. Summary:\n```\n"
               f"{json.dumps(done, indent=1)}\n```\nForward passes left: {self.budget - self.passes}.")
        return msg + (exp8.LAST_PASS if self.passes == self.budget else "")

    def close(self):
        self.kernel.stop()


def prepare():
    """Model files as plain files (the Hugging Face cache holds links, which a container mount cannot follow), MGSM
    split into practice and held-out problems, and the kernel's image."""
    import pandas as pd
    from huggingface_hub import hf_hub_download
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for f in ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"]:
        src, dst = Path(hf_hub_download(REPO, f, revision=REVISION)).resolve(), MODEL_DIR / f
        if not dst.exists():
            try:
                os.link(src, dst)   # no second copy of the 3.5 GB of weights
            except OSError:
                shutil.copy(src, dst)
    for folder in [PRACTICE, TEST]:
        folder.mkdir(parents=True, exist_ok=True)
    for lang in LANGS:
        df = pd.read_parquet(hf_hub_download(MGSM, f"{lang}/test-00000-of-00001.parquet", repo_type="dataset",
                                             revision=MGSM_REVISION))
        rows = [{"id": f"mgsm_{lang}_{i}", "source": f"mgsm_{lang}", "lang": lang, "problem": r.question,
                 "answer": float(r.answer_number)} for i, r in enumerate(df.itertuples())]
        for folder, part in [(PRACTICE, rows[:N_PRACTICE]), (TEST, rows[N_PRACTICE:])]:
            (folder / f"mgsm_{lang}.jsonl").write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in part), encoding="utf-8")
        print(f"mgsm_{lang}: {N_PRACTICE} practice, {len(rows) - N_PRACTICE} held out")
    subprocess.run(["docker", "build", "-q", "-t", IMAGE, "-f", str(ROOT / "sandbox" / "exp9.Dockerfile"),
                    str(ROOT / "sandbox")], check=True)
    print(f"ready: {CACHE}, image {IMAGE}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    r = sub.add_parser("run")
    r.add_argument("run", choices=list(BUDGETS))
    r.add_argument("--model", default="claude-opus-5-5")
    r.add_argument("--sample", type=int, default=1)
    a = ap.parse_args()
    if a.cmd == "prepare":
        prepare()
        sys.exit()
    from wsp.conversation import converse
    from wsp.runner import scrub_inherited_env
    if not (MODEL_DIR / "model.safetensors").exists():
        sys.exit("run `python -m wsp.exp9 prepare` first")
    scrub_inherited_env()
    converse(a.model, "exp9", a.run, a.sample, message(a.run), lambda out: Turns(a.run, out, a.sample))
