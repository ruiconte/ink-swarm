"""Read a rendered surface (sv.npy, layers along the mesh normals) with ink models in BOTH layer orders (page and reversed-order control).
From a 33-layer render along the mesh normals: follow the sheet pixel by pixel (local CT maximum, smoothed), read a
17-layer window with Argus V3 at 5 depths (-2..+2, averaged), and with reader-v3 and Nieuwlaar at depth 0 as
independent confirmations; every model also reads the reversed layer order (control). Same deconvolution + local
contrast and the same grey stretch for page and control. Fully programmatic, no hand editing.
usage: final4.py SV.npy OUT_DIR"""
import time; T0 = time.time(); import os, sys, json, types, numpy as np, torch
from PIL import Image
from scipy import ndimage
sys.path.insert(0, os.getcwd())
import train as T
from eval_all import load
from papyrus_deconv import pdeconv
sv = np.load(sys.argv[1], mmap_mode="r"); out = sys.argv[2]; os.makedirs(out, exist_ok=True)
L = sv.shape[0]; c0 = L // 2 + int(os.environ.get("CENTER", "0")); valid = np.asarray(sv[c0]) > 0; H, W = valid.shape
# per-pixel sheet depth (same rule as multiwrap.py, +-5 layers around the mesh)
ds = 8; sub = np.stack([ndimage.zoom(np.asarray(sv[j], np.float32), 1 / ds, order=1) for j in range(c0 - 5, c0 + 6)])
d = np.argmax(ndimage.gaussian_filter(sub, (1, 2, 2)), 0).astype(np.float32) - 5
d = ndimage.gaussian_filter(ndimage.median_filter(d, 9), 4); D = np.clip(ndimage.zoom(d, (H / d.shape[0], W / d.shape[1]), order=1)[:H, :W], -5, 5)
print("sheet depth offset: mean", round(float(D[valid].mean()), 2), "sd", round(float(D[valid].std()), 2), flush=True)
SVA = None
def _gather(z0):                                       # vectorised: value of layer z0 and z0+1 at every pixel (was a per-layer mask loop)
    global SVA
    if SVA is None: SVA = np.ascontiguousarray(sv)
    zmin, zmax = int(z0.min()), int(z0.max()) + 1; blk = SVA[zmin:zmax + 1]; k = (z0 - zmin)[None]
    return np.take_along_axis(blk, k, 0)[0].astype(np.float32), np.take_along_axis(blk, k + 1, 0)[0].astype(np.float32)
def window(off):
    win = np.zeros((17, H, W), np.uint8)
    for j in range(17):
        z = c0 - 8 + j + off + D; z0 = np.clip(np.floor(z).astype(int), 0, L - 2); f = (z - z0).astype(np.float32)
        lo, hi = _gather(z0)
        win[j] = np.clip((1 - f) * lo + f * hi, 0, 255)
    return win
dev = torch.device("cuda"); args = types.SimpleNamespace(patch=256, depth=17, batch=int(os.environ.get("PRED_BATCH", "24")), amp="bf16", tta=4, skip_empty=True)
class Seg(T.Segment):
    def __init__(s, v):
        s.vol, s.D = v, v.shape[0]; s.H, s.W = v.shape[1:]; s.sup = valid; s.ink = s.train_mask = np.zeros_like(valid)
PLAN = {"argus-v3": ("release/argus-v3/argus-v3.pth", (-2, -1, 0, 1, 2)), "reader-v3": ("official/reader-v3.pth", (0,)),
        "nieuwlaar": ("official/nieuwlaar-dense-native.safetensors", (0,))}
wins = {}
if os.environ.get("FINAL_MODELS"):                     # e.g. "argus-v3=release/argus-v3/argus-v3.pth,v4=runs/v4i-ctx-h1-med/last.pt"
    PLAN = {k: (v, (-2, -1, 0, 1, 2)) for k, v in (x.split("=", 1) for x in os.environ["FINAL_MODELS"].split(","))}
PLAN = {k: v for k, v in PLAN.items() if os.path.exists(v[0])}
if os.environ.get("FINAL_OFFS"):                       # screening: fewer depths (e.g. "0"), the deep read keeps -2..2
    PLAN = {k: (v[0], tuple(int(x) for x in os.environ["FINAL_OFFS"].split(","))) for k, v in PLAN.items()}
for name, (path, offs) in PLAN.items():
    f = f"{out}/p_{name}.npz"
    if os.path.exists(f): continue
    m = load(path)[0].to(dev).eval(); acc = {"fwd": 0, "rev": 0}
    for o in offs:
        w = wins.get(o)
        if w is None: w = wins[o] = window(o)
        for tag, vol in (("fwd", w), ("rev", w[::-1].copy())):
            acc[tag] = acc[tag] + np.where(valid, T.predict(m, Seg(vol), args, dev, region=valid), 0) / len(offs)
        print(name, "depth", o, "done", round(time.time() - T0), "s", flush=True)
    np.savez_compressed(f, fwd=acc["fwd"].astype(np.float16), rev=acc["rev"].astype(np.float16)); del m; torch.cuda.empty_cache()
v = ndimage.binary_fill_holes(ndimage.binary_closing(valid, iterations=5)); w = v.astype(np.float32)
def push(p, s1=20, s2=80):
    p = np.where(v, p, 0); m1 = ndimage.gaussian_filter(p, s1) / np.maximum(ndimage.gaussian_filter(w, s1), 1e-6)
    m2 = ndimage.gaussian_filter(p, s2) / np.maximum(ndimage.gaussian_filter(w, s2), 1e-6); return np.clip(p + 1.5 * (p - m1) + (p - m2), 0, 1)
summ = {}
for name in PLAN:
    z = np.load(f"{out}/p_{name}.npz"); P = {k: z[k].astype(np.float32) for k in ("fwd", "rev")}
    ink = {k: float((P[k][valid] > 0.5).mean()) for k in P}; rd = os.environ.get("READ", "rev") if name != "nieuwlaar" else max(ink, key=ink.get)
    if os.environ.get("READ") == "auto": rd = max(ink, key=ink.get)
    ct = "fwd" if rd == "rev" else "rev"
    da, dc = pdeconv(P[rd], v), pdeconv(P[ct], v); sc = np.percentile(da[v], 99.5); qa, qc = push(da / sc), push(dc / sc)
    lo, hi = np.percentile(qa[v], 1), np.percentile(qa[v], 99.5)
    for q, nm in ((qa, "page"), (qc, "ctl")):
        Image.fromarray(np.where(v, np.clip((q - lo) / (hi - lo + 1e-6), 0, 1) * 215 + 40, 0).astype(np.uint8)).save(f"{out}/{name}_{nm}.png")
    summ[name] = dict(read=rd, ink_page=ink[rd], ink_ctl=ink[ct], bright_page=float((qa[v] > 0.5).mean()), bright_ctl=float((qc[v] > 0.5).mean()))
    print(name, summ[name], flush=True)
# agreement between models on the page (independent confirmations) vs the same between controls
def z_(a): a = a[v]; return (a - a.mean()) / (a.std() + 1e-6)
pg = {n: np.asarray(Image.open(f"{out}/{n}_page.png"), np.float32) for n in PLAN}; cg = {n: np.asarray(Image.open(f"{out}/{n}_ctl.png"), np.float32) for n in PLAN}
for a, b in [(a, b) for i, a in enumerate(PLAN) for b in list(PLAN)[i + 1:]]:
    summ[f"corr {a}~{b}"] = dict(page=float((z_(pg[a]) * z_(pg[b])).mean()), ctl=float((z_(cg[a]) * z_(cg[b])).mean()))
    print(a, "~", b, summ[f"corr {a}~{b}"], flush=True)
json.dump(summ, open(f"{out}/summary.json", "w"), indent=1)
