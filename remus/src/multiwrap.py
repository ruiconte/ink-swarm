"""'Core sample' through the scroll: from a deep render along the normals of one surface (2R+1 layers), find every
sheet crossed (peaks of the mean CT profile), follow each sheet pixel by pixel (local CT maximum near its mean depth,
smoothed), resample a 17-layer window centred on it, and read it with Argus V3 in both layer orders.
Outputs per sheet: ink map, control map, and a summary ranked by map/control contrast.
usage: multiwrap.py DEEP.npy OUT_DIR"""
import json, os, sys, types, numpy as np, torch
from scipy import ndimage
from scipy.signal import find_peaks
import train as T
from eval_all import load
sv = np.load(sys.argv[1], mmap_mode="r"); out = sys.argv[2]; os.makedirs(out, exist_ok=True)
L = sv.shape[0]; c0 = L // 2; valid = np.asarray(sv[c0]) > 0
prof = np.array([float(np.asarray(sv[k])[valid].mean()) for k in range(L)])
pk, _ = find_peaks(ndimage.gaussian_filter1d(prof, 1.5), distance=7, prominence=0.4)
if os.environ.get("MW_STEP"):                       # sheets drift away from parallel far from the surface: sample every MW_STEP layers, per-pixel refinement does the rest
    st = int(os.environ["MW_STEP"]); pk = np.arange(c0 % st + 14, L - 15, st)
print("sheets at offsets", [int(k - c0) for k in pk], flush=True)
dev = torch.device("cuda"); args = types.SimpleNamespace(patch=256, depth=17, batch=8, amp="bf16", tta=4)
m = load("release/argus-v3/argus-v3.pth")[0].to(dev).eval()
class Seg(T.Segment):
    def __init__(s, v):
        s.vol, s.D = v, v.shape[0]; s.H, s.W = v.shape[1:]; s.sup = valid; s.ink = s.train_mask = np.zeros_like(valid)
H, W = valid.shape; ds = 8; summary = []
for k in pk:
    if k - 14 < 0 or k + 15 > L: continue
    # per-pixel depth of this sheet: argmax of smoothed CT within +-5 layers, on a coarse grid, smoothed
    sub = np.stack([ndimage.zoom(np.asarray(sv[j], np.float32), 1 / ds, order=1) for j in range(k - 5, k + 6)])
    sub = ndimage.gaussian_filter(sub, (1, 2, 2)); d = np.argmax(sub, 0).astype(np.float32) - 5
    d = ndimage.median_filter(d, 9); d = ndimage.gaussian_filter(d, 4)
    D = np.clip(ndimage.zoom(d, (H / d.shape[0], W / d.shape[1]), order=1)[:H, :W], -5, 5)
    yy, xx = np.mgrid[0:H, 0:W]
    win = np.zeros((17, H, W), np.uint8)
    for j in range(17):
        z = k - 8 + j + D; z0 = np.floor(z).astype(int); f = (z - z0).astype(np.float32)
        a = np.zeros((H, W), np.float32)
        for zz in np.unique(z0):
            sel = z0 == zz
            if 0 <= zz < L - 1:
                a[sel] = (1 - f[sel]) * np.asarray(sv[zz])[sel] + f[sel] * np.asarray(sv[zz + 1])[sel]
        win[j] = np.clip(a, 0, 255).astype(np.uint8)
    res = {}
    for tag, vol in (("fwd", win), ("rev", win[::-1].copy())):
        p = np.where(valid, T.predict(m, Seg(vol), args, dev, region=valid), 0); np.save(f"{out}/sheet{int(k - c0):+d}_{tag}.npy", p.astype(np.float16))
        res[tag] = float((p[valid] > 0.5).mean())
    hi, lo = max(res.values()), min(res.values())
    summary.append(dict(offset=int(k - c0), ink_fwd=res["fwd"], ink_rev=res["rev"], ratio=hi / max(lo, 1e-4), reading=max(res, key=res.get)))
    print(summary[-1], flush=True)
json.dump(sorted(summary, key=lambda s: -s["ratio"]), open(f"{out}/summary.json", "w"), indent=1)
