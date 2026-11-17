"""Argus vs public models on the native 9.4 um / 113 keV render of never-seen PHerc0841 w00 (reader-v3's home format).
usage: native_eval.py [--tta 8] [--patch 256]"""
import argparse, json, os, types
import numpy as np, torch
import train as T
from eval_all import load
from heldout_eval import Reversed
MODELS = {"Argus V1": "release/argus-v1/argus-v1.pth", "reader-v3": "official/reader-v3.pth",
          "dense-native": "official/nieuwlaar-dense-native.safetensors", "dense-all7": "official/dense-all7-75k.pth"}
import sys
ap = argparse.ArgumentParser(); ap.add_argument("--only", default=""); ap.add_argument("--tta", type=int, default=8); ap.add_argument("--patch", type=int, default=256)
ap.add_argument("--extra", nargs="*", default=[])
a = ap.parse_args()
for e in a.extra:
    k, v = e.split("=", 1); MODELS[k] = v
if a.only: MODELS = {k: v for k, v in MODELS.items() if k in a.only.split(",")}
dev = torch.device("cuda")
args = types.SimpleNamespace(patch=a.patch, depth=17, batch=8 if a.patch > 128 else 32, amp="bf16", tta=a.tta)
s = T.Segment("pherc0841-w00-native"); rev = Reversed(s)
lab, reg = s.ink_eval, s.val
res = {}
for n, p in MODELS.items():
    m = load(p)[0].to(dev).eval()
    pr = T.predict(m, s, args, dev, region=reg)
    met = T.metrics(pr, lab, reg, 0.5)
    t80 = float(np.quantile(pr[reg & lab], 0.2)); bg = float((pr[reg & ~lab] >= t80).mean())
    mr = T.metrics(T.predict(m, rev, args, dev, region=reg), lab, reg, 0.5)
    res[n] = dict(auc=met["auc"], bacc=met["bacc@0.5"], dice=met["dice_best"], bg80=bg, auc_rev=mr["auc"])
    print(f"{n:13s} AUC {met['auc']:.4f} bAcc {met['bacc@0.5']:.4f} Dice {met['dice_best']:.4f} bg@80 {bg*100:.1f}% rev AUC {mr['auc']:.4f}", flush=True)
    del m; torch.cuda.empty_cache()
old = json.load(open("runs/native_0841_results.json")) if os.path.exists("runs/native_0841_results.json") else {}
json.dump({**old, **res}, open("runs/native_0841_results.json", "w"), indent=1)
