"""Contrast-free comparison Argus V3 vs Reader V3 (answer to the Discord remark "it's just more contrast").
Per segment: both maps rank-normalised inside the papyrus (identical histograms, so any brightness/contrast tweak is
undone), then scored against the experts' labels with measures that ignore contrast:
  AUC, precision at the true ink fraction (top-k pixels, k = labelled ink amount), and where the two disagree.
Writes runs/fair_compare.json and runs/fair_<seg>.png (rank-normalised maps side by side + disagreement).
usage: fair_compare.py [seg ...]   (dev segments only, never data/sealed-*)"""
import json, sys, types
import numpy as np, torch
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import roc_auc_score
import train as T
from eval_all import load

SEGS = sys.argv[1:] or ["pherc0841-w00-native", "pherc0139-w016", "pherc0814-46527", "pherc1667-w029"]
assert not any(s.startswith("sealed") for s in SEGS)
dev = torch.device("cuda"); args = types.SimpleNamespace(patch=256, depth=17, batch=8, amp="bf16", tta=4)
M = {n: load(p)[0].to(dev).eval() for n, p in (("Argus V3", "release/argus-v3/argus-v3.pth"), ("Reader V3", "official/reader-v3.pth"))}


def ranknorm(p, m):
    out = np.zeros_like(p, np.float32); v = p[m]
    out[m] = (np.argsort(np.argsort(v, kind="stable"), kind="stable") + 0.5) / len(v); return out


res = {}
for seg in SEGS:
    s = T.Segment(seg); m = s.val if s.val.any() else s.sup; y = s.ink_eval[m]
    P = {n: T.predict(mod, s, args, dev, region=m) for n, mod in M.items()}
    R = {n: ranknorm(p, m) for n, p in P.items()}
    k = int(y.sum()); r = {}
    for n in M:
        v = P[n][m]; top = np.argsort(-v)[:k]
        r[n] = dict(auc=round(float(roc_auc_score(y, v)), 4), prec_at_true_ink=round(float(y[top].mean()), 4))
    A, B = R["Argus V3"][m], R["Reader V3"][m]
    thr = 1 - y.mean()                                   # each map: its own top-k pixels = "ink"
    a_only, b_only = (A >= thr) & (B < thr), (B >= thr) & (A < thr)
    r["disagree"] = dict(argus_only_px=int(a_only.sum()), argus_only_true=round(float(y[a_only].mean()), 4),
                         reader_only_px=int(b_only.sum()), reader_only_true=round(float(y[b_only].mean()), 4),
                         corr_rank=round(float(np.corrcoef(A, B)[0, 1]), 4))
    res[seg] = r; print(seg, json.dumps(r, ensure_ascii=False), flush=True)
    # picture: rank-normalised maps + labels + disagreement (green = Argus only, magenta = Reader only, white = both)
    ys, xs = np.nonzero(m); y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    def g(a): return Image.fromarray((a[y0:y1, x0:x1] * 255).astype(np.uint8)).convert("RGB")
    Ab, Bb = R["Argus V3"] >= thr, R["Reader V3"] >= thr
    dis = np.zeros(m.shape + (3,), np.uint8)
    dis[Ab & Bb] = 255; dis[Ab & ~Bb] = (60, 220, 60); dis[Bb & ~Ab] = (230, 60, 230)
    tiles = [("Argus V3 (rank-normalised)", g(R["Argus V3"])), ("Reader V3 (rank-normalised)", g(R["Reader V3"])),
             ("expert labels", g(s.ink_eval.astype(np.float32))), ("green=Argus only  magenta=Reader only", Image.fromarray(dis[y0:y1, x0:x1]))]
    w, h = tiles[0][1].size; sc = min(1.0, 1400 / w); w2, h2 = int(w * sc), int(h * sc)
    out = Image.new("RGB", (w2, (h2 + 50) * 4), "white"); f = ImageFont.truetype("arial.ttf", 34); d = ImageDraw.Draw(out)
    for i, (t, im) in enumerate(tiles):
        out.paste(im.resize((w2, h2)), (0, i * (h2 + 50) + 50)); d.text((8, i * (h2 + 50) + 6), t, fill=(0, 0, 0), font=f)
    out.save(f"runs/fair_{seg}.png")
json.dump(res, open("runs/fair_compare.json", "w"), indent=1)
