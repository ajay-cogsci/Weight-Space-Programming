"""exp9's forward-pass tool and evaluation, run on this machine with transformers (CUDA, Apple MPS or CPU): the local
counterpart of exp8_forward.py and exp8_eval.py. The model stays loaded between passes; the edits (/work/edited/*.npy,
written by the kernel) are put into it and taken out again in place.

  python -m wsp.exp9_forward eval <edit folder | none> <out folder> [--n 150] [--langs ja,en,zh]

eval: the held-out problems (never mounted in the kernel), one reply each, unedited and edited with the same seeds;
writes <out folder>/{original,edited}.jsonl and score.json.
"""
from __future__ import annotations

import json
import os
import sys
import time
import zlib
from collections import Counter
from pathlib import Path

from wsp.exp9_lang import grade

CACHE = Path.home() / ".cache" / "wsp" / "exp9"
MODEL_DIR, PRACTICE, TEST = CACHE / "model", CACHE / "practice", CACHE / "test"
FROZEN = {"model.norm.weight", "lm_head.weight", "model.embed_tokens.weight"}
LAYERS, WIDTH, MLP_WIDTH, HEADS = 28, 1536, 8960, 12
SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 50}   # the model card's, and transformers' default top_k
EVAL_TOKENS, BATCH = 2048, int(os.environ.get("WSP_BATCH", 8))   # replies generated at once: 8 fits a laptop
LIMITS = {"gen_max": 128, "gen_tokens": EVAL_TOKENS, "tf_max_texts": 256, "tf_max_tokens": 4096,
          "tf_total_tokens": 65536, "act_max_tokens": 2048, "attn_max_texts": 2, "attn_max_tokens": 1024,
          "max_bytes": 2 << 30}
_NET = None


class Net:
    """The model and its tokenizer; use() switches between the original weights and an edit."""

    def __init__(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        dtype = self.dtype = {"cuda": torch.bfloat16, "mps": torch.float16, "cpu": torch.float32}[self.device]
        self.tok = AutoTokenizer.from_pretrained(MODEL_DIR, padding_side="left")
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=dtype).to(self.device).eval()
        self.params = dict(self.model.named_parameters())
        self.saved = {}   # name -> original values of the tensors that are edited now

    def use(self, which: str, edits: Path | None) -> list[str]:
        """Put the original weights back and, for "edited", the tensors of `edits` in; the edited tensors' names.
        Checks what save_edit checks, since a file can be written to the folder without it."""
        import numpy as np
        import torch
        for name, t in self.saved.items():
            self.params[name].data.copy_(t)
        self.saved = {}
        if which != "edited" or edits is None or not edits.is_dir():
            return []
        new = {}
        for f in sorted(edits.glob("*.npy")):
            name, t = f.name[:-4], torch.from_numpy(np.load(f))
            if name in FROZEN or name not in self.params:
                raise ValueError(f"edited/{f.name}: {name} may not be edited" if name in FROZEN
                                 else f"edited/{f.name}: no tensor named {name}")
            if tuple(t.shape) != tuple(self.params[name].shape) or not torch.isfinite(t).all():
                raise ValueError(f"edited/{f.name}: wrong shape, or inf or nan values")
            new[name] = t
        for name, t in new.items():
            p = self.params[name]
            self.saved[name] = p.data.clone()
            p.data.copy_(t.to(p.dtype))
        return sorted(new)

    def attention(self, eager: bool):
        """Attention probabilities are only returned by the "eager" implementation, which gives nan in float16 on
        MPS: such a pass runs in float32 (half precision to float32 and back is exact)."""
        import torch
        self.model.set_attn_implementation("eager" if eager else "sdpa")
        self.model.to(torch.float32 if eager else self.dtype)


def net() -> Net:
    global _NET
    if _NET is None:
        _NET = Net()
    return _NET


def render(tok, user: str) -> str:
    """The evaluation's prompt: the chat template around the user message, ending with "<think>\\n"."""
    p = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
    return p if p.endswith("<think>\n") else p + "<think>\n"


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _items(req: dict, practice: dict) -> list[dict]:
    rows = []
    for pid in req.get("problems", []):
        if pid not in practice:
            raise ValueError(f"unknown practice id: {pid}")
        rows.append({**practice[pid], "user": practice[pid]["problem"]})
    rows += [{"id": f"prompt_{i}", "source": "prompt", "lang": req.get("prompt_lang"), "user": p}
             for i, p in enumerate(req.get("prompts", []))]
    return rows


def _generate(nt: Net, rows: list[dict], n: int, max_tokens: int) -> tuple[list[dict], dict]:
    """n replies per row, sampled with SAMPLING and graded by exp9_lang.grade. Longest prompts first, so a batch pads
    little; a batch is seeded by its first item, so the same request gives both models the same seeds."""
    import torch
    tok, model = nt.tok, nt.model
    prompts = [render(tok, r["user"]) for r in rows]
    jobs = [(i, j) for i in range(len(rows)) for j in range(n)]
    order = sorted(range(len(jobs)), key=lambda k: -len(prompts[jobs[k][0]]))
    out = [None] * len(jobs)
    for b in range(0, len(order), BATCH):
        idx = order[b:b + BATCH]
        enc = tok([prompts[jobs[k][0]] for k in idx], return_tensors="pt", padding=True, add_special_tokens=False)
        enc = enc.to(model.device)
        first = jobs[idx[0]]
        torch.manual_seed(zlib.crc32(f"{rows[first[0]]['id']}/{first[1]}".encode()))
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_tokens, do_sample=True, pad_token_id=tok.pad_token_id,
                                 **SAMPLING)
        for k, row in zip(idx, gen[:, enc.input_ids.shape[1]:].tolist()):
            stops = [p for p, t in enumerate(row) if t == tok.eos_token_id]
            out[k] = (row[:stops[0]] if stops else row, not stops and len(row) >= max_tokens)
        print(f"  generated {min(b + BATCH, len(order))}/{len(order)}", file=sys.stderr, flush=True)
    res = []
    for i, r in enumerate(rows):
        samples = []
        for j in range(n):
            ids, cut = out[i * n + j]
            text = tok.decode(ids, skip_special_tokens=True)
            samples.append({"text": text, "tokens": len(ids), "cut": cut, **grade(text, r.get("lang"), r.get("answer")),
                            "_ids": ids})
        res.append({"id": r["id"], "source": r["source"], "lang": r.get("lang"), "answer": r.get("answer"),
                    "samples": samples, "_prompt_ids": tok(prompts[i], add_special_tokens=False).input_ids})
    return res, _summary(res)


def _rate(values) -> float | None:
    v = [x for x in values if x is not None]
    return round(sum(v) / len(v), 4) if v else None


def _summary(res: list[dict]) -> dict:
    def one(ss):
        return {"generations": len(ss), "think_in_lang": _rate(s["think_in_lang"] for s in ss),
                "answer_in_lang": _rate(s["answer_in_lang"] for s in ss), "accuracy": _rate(s["correct"] for s in ss),
                "think_lang": dict(Counter(s["think_lang"] for s in ss).most_common()),
                "closed_think": _rate(s["closed_think"] for s in ss), "cut_share": _rate(s["cut"] for s in ss),
                "mean_tokens": int(sum(s["tokens"] for s in ss) / len(ss))}
    by = {}
    for x in res:
        by.setdefault(x["source"], []).extend(x["samples"])
    return {**one([s for x in res for s in x["samples"]]), "by_source": {k: one(v) for k, v in sorted(by.items())}}


def _paired(ro: list[dict], re_: list[dict]) -> dict:
    """Edited minus original per item, by source: the mean difference and its standard error."""
    out = {}
    for src in sorted({x["source"] for x in ro}):
        out[src] = {}
        for key in ["think_in_lang", "answer_in_lang", "correct"]:
            d = []
            for a, b in zip(ro, re_):
                va, vb = [s[key] for s in a["samples"]], [s[key] for s in b["samples"]]
                if a["source"] == src and None not in va + vb:
                    d.append(sum(vb) / len(vb) - sum(va) / len(va))
            if not d:
                continue
            mean = sum(d) / len(d)
            se = (sum((x - mean) ** 2 for x in d) / (len(d) - 1)) ** 0.5 / len(d) ** 0.5 if len(d) > 1 else None
            out[src][key] = {"items": len(d), "difference": round(mean, 4),
                             "standard_error": round(se, 4) if se is not None else None}
    return out


def _write(path: Path, res: list[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for x in res:
            row = {k: v for k, v in x.items() if k != "_prompt_ids"}
            row["samples"] = [{a: b for a, b in s.items() if a != "_ids"} for s in x["samples"]]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _capture(model, layers: list[int], mlp: list[int], attn: list[int]):
    """Hooks: residual stream after layer l (0 = embeddings), down_proj inputs, attention probabilities."""
    got, hooks = {}, []
    base = model.model
    if 0 in layers:
        hooks.append(base.embed_tokens.register_forward_hook(lambda m, i, o: got.__setitem__("hidden_0", o[0])))
    for layer in layers:
        if layer > 0:
            hooks.append(base.layers[layer - 1].register_forward_hook(
                lambda m, i, o, layer=layer: got.__setitem__(f"hidden_{layer}",
                                                             (o[0] if isinstance(o, tuple) else o)[0])))
    for layer in mlp:
        hooks.append(base.layers[layer].mlp.down_proj.register_forward_pre_hook(
            lambda m, i, layer=layer: got.__setitem__(f"mlp_act_{layer}", i[0][0])))
    for layer in attn:
        hooks.append(base.layers[layer].self_attn.register_forward_hook(
            lambda m, i, o, layer=layer: got.__setitem__(f"attn_{layer}", o[1][0])))
    return got, hooks


def _layers(values, top: int, what: str) -> list[int]:
    values = [int(v) for v in values]
    if any(not 0 <= v <= top for v in values):
        raise ValueError(f"{what}: layer indices 0..{top}")
    return values


def _activations(nt: Net, res: list[dict], layers: list[int], mlp: bool, dst: Path) -> dict:
    """Per generated reply: residual stream (and down_proj inputs) at the last prompt token and averaged over the
    reply's first act_max_tokens tokens."""
    import numpy as np
    import torch
    mlp_layers = [layer for layer in layers if layer < LAYERS] if mlp else []
    got, hooks = _capture(nt.model, layers, mlp_layers, [])
    hl, ha, ml, ma, index = [], [], [], [], []
    with torch.no_grad():
        for x in res:
            p = x["_prompt_ids"]
            for j, s in enumerate(x["samples"]):
                a = s["_ids"][:LIMITS["act_max_tokens"]]
                if not a:
                    continue
                got.clear()
                nt.model.model(torch.tensor([p + a], device=nt.device))
                cut = len(p) - 1
                hl.append(torch.stack([got[f"hidden_{layer}"][cut] for layer in layers]).float().cpu())
                ha.append(torch.stack([got[f"hidden_{layer}"][len(p):].float().mean(0) for layer in layers]).cpu())
                if mlp_layers:
                    ml.append(torch.stack([got[f"mlp_act_{layer}"][cut] for layer in mlp_layers]).float().cpu())
                    ma.append(torch.stack([got[f"mlp_act_{layer}"][len(p):].float().mean(0)
                                           for layer in mlp_layers]).cpu())
                index.append({"item": x["id"], "sample": j, "tokens": len(a),
                              **{k: s[k] for k in ["think_lang", "answer_lang", "think_in_lang", "correct"]}})
    for h in hooks:
        h.remove()
    dst.mkdir(parents=True, exist_ok=True)
    arrays = {"hidden_last_prompt": hl, "hidden_mean_answer": ha}
    if mlp_layers:
        arrays.update({"mlp_last_prompt": ml, "mlp_mean_answer": ma})
    shapes = {}
    for name, rows in arrays.items():
        a = torch.stack(rows).numpy().astype(np.float16)
        np.save(dst / f"{name}.npy", a)
        shapes[f"{name}.npy"] = list(a.shape)
    (dst / "index.json").write_text(json.dumps({"layers": layers, "mlp_layers": mlp_layers, "samples": index}))
    return shapes


def _teacher_forced(nt: Net, req: dict, dst: Path, edited: list[str]) -> dict:
    import numpy as np
    import torch
    tok, model = nt.tok, nt.model
    texts = list(req.get("texts", []))
    ret = req.get("return", {})
    ids = [tok(t, add_special_tokens=False)["input_ids"] for t in texts]
    lengths = [len(i) for i in ids]
    if not 0 < len(ids) <= LIMITS["tf_max_texts"] or min(lengths) < 1:
        raise ValueError(f"teacher_forced takes 1 to {LIMITS['tf_max_texts']} non-empty texts")
    if max(lengths) > LIMITS["tf_max_tokens"] or sum(lengths) > LIMITS["tf_total_tokens"]:
        raise ValueError(f"texts are limited to {LIMITS['tf_max_tokens']} tokens each, "
                         f"{LIMITS['tf_total_tokens']} in total")
    layers = _layers(ret.get("hidden_states", []), LAYERS, "hidden_states")
    mlp = _layers(ret.get("mlp_act", []), LAYERS - 1, "mlp_act")
    attn = _layers(ret.get("attn_layers", []), LAYERS - 1, "attn_layers")
    if attn and (len(ids) > LIMITS["attn_max_texts"] or max(lengths) > LIMITS["attn_max_tokens"]):
        raise ValueError(f"attn_layers: at most {LIMITS['attn_max_texts']} texts of at most "
                         f"{LIMITS['attn_max_tokens']} tokens")
    topk = int(ret.get("topk", 0))
    if topk > 50:
        raise ValueError("topk is at most 50")
    T, L = len(ids), max(lengths)
    shapes = {"input_ids": (T, L)}
    if ret.get("token_logprobs"):
        shapes["token_logprobs"] = (T, L - 1)
    if topk:
        shapes["topk_logprobs"] = shapes["topk_ids"] = (T, L, topk)
    shapes.update({f"hidden_{layer}": (T, L, WIDTH) for layer in layers})
    shapes.update({f"mlp_act_{layer}": (T, L, MLP_WIDTH) for layer in mlp})
    shapes.update({f"attn_{layer}": (T, HEADS, L, L) for layer in attn})
    size = sum(int(np.prod(s)) * (8 if k in ("input_ids", "topk_ids") else 4 if "logprobs" in k else 2)
               for k, s in shapes.items())
    if size > LIMITS["max_bytes"]:
        raise ValueError(f"the results would take {size / 2**30:.1f} GB; at most "
                         f"{LIMITS['max_bytes'] / 2**30:.0f} GB per pass")
    dst.mkdir(parents=True, exist_ok=True)
    dtypes = {"input_ids": np.int64, "topk_ids": np.int64, "token_logprobs": np.float32, "topk_logprobs": np.float32}
    arrs = {k: np.lib.format.open_memmap(dst / f"{k}.npy", "w+", dtypes.get(k, np.float16), s)
            for k, s in shapes.items()}
    got, hooks = _capture(model, layers, mlp, attn)
    mean_lp = []
    nt.attention(eager=bool(attn))
    try:
        with torch.no_grad():
            for i, x in enumerate(ids):
                got.clear()
                t = torch.tensor([x], device=nt.device)
                h = model.model(t, output_attentions=bool(attn)).last_hidden_state[0]
                arrs["input_ids"][i, :len(x)] = x
                tl = []
                for a in range(0, len(x), 512):   # the 151,936-token vocabulary: logits in chunks
                    lp = torch.log_softmax(model.lm_head(h[a:a + 512]).float(), -1)
                    nxt = t[0, a + 1:a + 513]
                    tl.append(lp[:len(nxt)].gather(1, nxt[:, None])[:, 0])
                    if topk:
                        v, j = lp.topk(topk, -1)
                        arrs["topk_logprobs"][i, a:a + len(lp)] = v.cpu().numpy()
                        arrs["topk_ids"][i, a:a + len(lp)] = j.cpu().numpy()
                tl = torch.cat(tl)
                mean_lp.append(round(float(tl.mean()), 4) if len(x) > 1 else None)
                if "token_logprobs" in arrs:
                    arrs["token_logprobs"][i, :len(x) - 1] = tl.cpu().numpy()
                for k, v in got.items():
                    if k.startswith("attn_"):
                        arrs[k][i, :, :len(x), :len(x)] = v.float().cpu().numpy()
                    else:
                        arrs[k][i, :len(x)] = v.float().cpu().numpy()
    finally:
        nt.attention(eager=False)
        for h in hooks:
            h.remove()
    for a in arrs.values():
        a.flush()
    files = {f"{k}.npy": list(s) for k, s in shapes.items()}
    meta = {"texts": texts, "lengths": lengths, "model": req.get("model", "original"), "edited_tensors": edited,
            "files": files, "padding": "right; positions >= length are padding"}
    (dst / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return {"lengths": lengths, "files": files, "edited_tensors": len(edited), "mean_token_logprob": mean_lp}


def _forward(work: Path, k: int, req: dict) -> dict:
    t0 = time.time()
    kind, which = req.get("type"), req.get("model", "original")
    if kind not in ("generate", "teacher_forced"):
        raise ValueError('type must be "generate" or "teacher_forced"')
    if which not in ("original", "edited", "both") or (kind == "teacher_forced" and which == "both"):
        raise ValueError(f"model must be original or edited{' or both' if kind == 'generate' else ''}")
    nt = net()
    if kind == "teacher_forced":
        edited = nt.use(which, work / "edited")
        done = {"ok": True, **_teacher_forced(nt, req, work / f"fwd_{k}", edited), "type": "teacher_forced"}
    else:
        practice = {r["id"]: r for f in sorted(PRACTICE.glob("*.jsonl")) for r in read_jsonl(f)}
        rows = _items(req, practice)
        n, mt = int(req.get("n", 1)), int(req.get("max_tokens", LIMITS["gen_tokens"]))
        cap = LIMITS["gen_max"] // (2 if which == "both" else 1)
        if not rows or n < 1 or len(rows) * n > cap or not 1 <= mt <= LIMITS["gen_tokens"]:
            raise ValueError(f"items x n must be 1 to {cap}{' per model' if which == 'both' else ''}; "
                             f"max_tokens at most {LIMITS['gen_tokens']}")
        act = req.get("activations")
        layers = _layers(act.get("layers", []), LAYERS, "activations.layers") if act else []
        if act and not layers:
            raise ValueError("activations needs layers")
        out, files, acts = {}, {}, {}
        for m in ["original", "edited"] if which == "both" else [which]:
            nt.use(m, work / "edited")
            res, summ = _generate(nt, rows, n, mt)
            name = f"gen_{k}_{m}" if which == "both" else f"gen_{k}"
            _write(work / f"{name}.jsonl", res)
            if act:
                shapes = _activations(nt, res, layers, bool(act.get("mlp")), work / f"{name}_acts")
                acts[m] = {"dir": f"/work/{name}_acts/", "files": shapes}
            out[m], files[m] = (res, summ), f"/work/{name}.jsonl"
        if which == "both":
            done = {"ok": True, "type": "generate", "model": "both", "files": files, "activations": acts or None,
                    "original": out["original"][1], "edited": out["edited"][1],
                    "paired_edited_minus_original": _paired(out["original"][0], out["edited"][0])}
        else:
            done = {"activations": acts.get(which), "ok": True, "type": "generate", "model": which,
                    "file": files[which], **out[which][1]}
    nt.use("original", None)
    done["s"] = round(time.time() - t0, 1)
    return done


def run_forward(work: Path, k: int, request: dict) -> dict:
    """One forward pass for the run whose /work is `work`; a failure is reported to the model and not counted."""
    try:
        return _forward(Path(work), k, request)
    except Exception as e:
        if _NET is not None:
            _NET.use("original", None)
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def evaluate(edit: Path | None, out: Path, n: int, langs: list[str]) -> dict:
    rows = [{**r, "user": r["problem"]} for lang in langs for r in read_jsonl(TEST / f"mgsm_{lang}.jsonl")[:n]]
    out.mkdir(parents=True, exist_ok=True)
    nt, res, score = net(), {}, {"problems_per_language": n, "max_tokens": EVAL_TOKENS, "sampling": SAMPLING}
    for m in ["original"] + (["edited"] if edit else []):
        print(f"{m}: {len(rows)} replies", file=sys.stderr, flush=True)
        score[f"{m}_tensors"] = nt.use(m, edit)
        res[m], score[m] = _generate(nt, rows, 1, EVAL_TOKENS)
        _write(out / f"{m}.jsonl", res[m])
    if edit:
        score["paired_edited_minus_original"] = _paired(res["original"], res["edited"])
    (out / "score.json").write_text(json.dumps(score, indent=1), encoding="utf-8")
    return score


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["eval"])
    ap.add_argument("edit", help='folder of <tensor name>.npy files (a run\'s work/edited), or "none"')
    ap.add_argument("out", type=Path)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--langs", type=lambda s: s.split(","), default=["ja", "en", "zh"])
    a = ap.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    print(json.dumps(evaluate(None if a.edit == "none" else Path(a.edit), a.out, a.n, a.langs), indent=1))
