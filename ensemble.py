"""Argus V1 candidates: score single models and ensembles (mean of probabilities) on the three validation masks,
same harness as eval_all.py. Two checkpoint rules per run: best.pt (picked on these masks: optimistic) and last.pt
(end of training: chosen blind). usage: ensemble.py run1 run2 ... [--tta 8]"""
import argparse, itertools, json, os, types
import numpy as np
import torch

import official
import train as T

HERE = os.path.dirname(os.path.abspath(__file__))


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    a = ck.get("args", {})
    if a.get("arch", "official") == "official":
        m = official.OfficialInkNet("attn_max")
    else:
        ar = types.SimpleNamespace(**a)
        m = T.InkNet(ar.depth, ar.stem, ar.width, ar.levels, ar.zproj, getattr(ar, "stem_blocks", 0))
    m.load_state_dict(ck["model"])
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--tta", type=int, default=1)
    ap.add_argument("--out", default="runs/ensemble_results.json")
    a = ap.parse_args()
    dev = torch.device("cuda")
    segs = [T.Segment(n) for n in T.VAL_SEGMENTS]
    args = types.SimpleNamespace(patch=128, depth=17, batch=32, amp="bf16", tta=a.tta)
    probs = {}
    for run in a.runs:
        for kind in ("best", "last"):
            p = os.path.join(HERE, "runs", run, f"{kind}.pt")
            if not os.path.exists(p):
                continue
            m = load(p).to(dev).eval()
            probs[(run, kind)] = {s.name: T.predict(m, s, args, dev, region=s.val) for s in segs}
            del m
            torch.cuda.empty_cache()

    def score(keys):
        per = {s.name: T.metrics(np.mean([probs[k][s.name] for k in keys], 0), s.ink_eval, s.val, 0.5) for s in segs}
        return {m: round(float(np.mean([per[n][m] for n in per])), 4) for m in ("auc", "dice_best", "bacc@0.5")}

    res = {}
    for kind in ("best", "last"):
        keys = [k for k in probs if k[1] == kind]
        for k in keys:
            res[f"{kind} {k[0]}"] = score([k])
        if len(keys) > 1:
            res[f"{kind} ENSEMBLE all {len(keys)}"] = score(keys)
            singles = sorted(keys, key=lambda k: -res[f"{kind} {k[0]}"]["bacc@0.5"])
            for n in range(2, len(keys)):
                res[f"{kind} ENSEMBLE top{n} " + "+".join(k[0] for k in singles[:n])] = score(singles[:n])
    json.dump(res, open(os.path.join(HERE, a.out), "w"), indent=1)
    for k, v in res.items():
        print(f"{k[:80]:80s} AUC {v['auc']:.4f}  Dice {v['dice_best']:.4f}  bAcc {v['bacc@0.5']:.4f}", flush=True)


if __name__ == "__main__":
    main()
