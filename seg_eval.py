"""Score checkpoints on any prepared labelled segments (AUC, bAcc@0.5, best Dice, background at 80% recall),
optionally with a depth-window ensemble. usage: seg_eval.py --segs a,b --offs 0 name=path ..."""
import argparse, json, os, types, numpy as np, torch
import train as T
from eval_all import load
ap = argparse.ArgumentParser(); ap.add_argument("--segs", required=True); ap.add_argument("--offs", default="0")
ap.add_argument("--tta", type=int, default=2); ap.add_argument("--patch", type=int, default=256); ap.add_argument("models", nargs="+")
a = ap.parse_args()
dev = torch.device("cuda"); offs = [int(x) for x in a.offs.split(",")]
args = types.SimpleNamespace(patch=a.patch, depth=17, batch=8 if a.patch > 128 else 32, amp="bf16", tta=a.tta)
class Sub(T.Segment):
    def __init__(self, base, full, o):
        self.__dict__.update(base.__dict__); c = (base.D - 17) // 2; self.vol = full[c + o:c + o + 17]; self.D = 17
segs = []
for n in a.segs.split(","):
    s = T.Segment(n); segs.append((s, np.asarray(s.vol)))
for spec in a.models:
    name, path = spec.split("=", 1); m = load(path)[0].to(dev).eval(); out = []
    for s, full in segs:
        pr = np.mean([T.predict(m, Sub(s, full, o), args, dev, region=s.val) for o in offs], 0)
        met = T.metrics(pr, s.ink_eval, s.val, 0.5); t = float(np.quantile(pr[s.val & s.ink_eval], 0.2))
        out.append(f"{s.name[-22:]}: AUC {met['auc']:.4f} bAcc {met['bacc@0.5']:.4f} Dice {met['dice_best']:.4f} bg {(pr[s.val & ~s.ink_eval] >= t).mean()*100:.1f}%")
    print(f"{name:12s} " + " | ".join(out), flush=True)
    del m; torch.cuda.empty_cache()
