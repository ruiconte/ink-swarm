"""Ruico's whole-stack rule: analyse ALL sheets of a core at once, both directions. Model, per pixel column through
the stack: observed ink on sheet k  y_k = sum_s x_s * d**|k - s|, where x_s >= 0 is the ink really written on sheet s
and d is how much ink echoes onto the next sheet (each further sheet another factor d). A trend growing towards one
end points to a source there; several sources per column are allowed. Solved for every pixel at once (projected
gradient, non-negative). d is MEASURED: excess of shared ink between neighbouring sheets over chance (6 mm shift).
Per sheet: legit ink = x_k (displayed as y_k scaled), removed = echo it received, given back = echoes it emitted.
usage: stack_unmix.py CORE_DIR OUT_PREFIX [d]"""
import sys, os, json, numpy as np
from scipy import ndimage
from PIL import Image, ImageDraw
D, out = sys.argv[1], sys.argv[2]
S = json.load(open(os.path.join(D, "summary.json"))); by = {s["offset"]: s for s in S}
offs = sorted(by); real = [offs[0]]
for o in offs[1:]:
    if o - real[-1] >= 12: real.append(o)                                     # real sheets only (>= 12 layers apart)
Y = np.stack([np.load(os.path.join(D, f"sheet{o:+d}_{by[o]['reading']}.npy")).astype(np.float32) for o in real])
K, H, W = Y.shape; v = (Y != 0).all(0)
Yb = np.stack([ndimage.gaussian_filter(y, 6) for y in Y]) * v                 # tolerate small misregistration
BG = np.array([np.median(Yb[k][v]) for k in range(K)], np.float32)[:, None, None]   # papyrus background level (not ink): never moved
Ye = np.maximum(Yb - BG, 0) * v
def excess(a, b):
    ai, bi = a > 0.5, b > 0.5; q = (ai & bi).sum() / max(ai.sum(), 1); c = (ai & np.roll(np.roll(bi, 640, 0), 640, 1)).sum() / max(ai.sum(), 1)
    return q, c
if len(sys.argv) > 3: d = float(sys.argv[3])
else:                                                                          # d from the data: how strongly sheets echo their neighbours beyond chance
    ex = []
    for k in range(K - 1):
        q, c = excess(Yb[k] * v, Yb[k + 1] * v); ex.append(max(0.0, (q - c) / max(1 - c, 1e-6)))
    d = float(np.clip(np.median(ex), 0.0, 0.6))
print("sheets", real, "echo factor d (overlap count) =", round(d, 3), flush=True)
A = d ** np.abs(np.subtract.outer(np.arange(K), np.arange(K))).astype(np.float32)   # K x K mixing matrix
if os.environ.get("D_MODE", "intensity") == "intensity":
    # Ruico: measure HOW MUCH ink echoes from the INTENSITY of repeated motifs, per pair of sheets. In every 3 mm window
    # where the letter-scale shapes of two sheets match beyond chance (NCC above the 99th pct of a 6 mm shifted null),
    # the echo strength is the median ratio weaker/stronger of their ink excess at co-located ink pixels.
    bp = lambda a: ndimage.gaussian_filter(a, 6) - ndimage.gaussian_filter(a, 40)
    Wn, St = 320, 160; BPs = [bp(Ye[k]) for k in range(K)]
    def ncc(a, b):
        a = a - a.mean(); b = b - b.mean(); dd = a.std() * b.std(); return float((a * b).mean() / dd) if dd > 1e-6 else 0.0
    Dm = np.eye(K, dtype=np.float32); LOCAL = {}
    for i in range(K):
        for j in range(i + 1, K):
            sims, nulls, ratios = [], [], []
            sh = np.roll(np.roll(BPs[j], 640, 0), 640, 1)
            for y0 in range(0, H - Wn, St):
                for x0 in range(0, W - Wn, St):
                    if v[y0:y0 + Wn, x0:x0 + Wn].mean() < 0.95: continue
                    a, b = BPs[i][y0:y0 + Wn, x0:x0 + Wn], BPs[j][y0:y0 + Wn, x0:x0 + Wn]
                    sims.append((ncc(a, b), y0, x0)); nulls.append(ncc(a, sh[y0:y0 + Wn, x0:x0 + Wn]))
            if len(nulls) < 10: continue
            thr = max(0.4, float(np.percentile(nulls, 99)))
            for r, y0, x0 in sims:
                if r <= thr: continue
                ei, ej = Ye[i][y0:y0 + Wn, x0:x0 + Wn], Ye[j][y0:y0 + Wn, x0:x0 + Wn]; m = (ei > 0.15) & (ej > 0.15)
                if m.sum() > 200:
                    rr = float(np.median(np.minimum(ei[m], ej[m]) / np.maximum(ei[m], ej[m]))); ratios.append(rr)
                    LOCAL.setdefault((i, j), []).append((y0, x0, rr))
            share = sum(1 for r, _, _ in sims if r > thr) / max(len(sims), 1); expect = 0.01
            w = max(0.0, (share - expect) / max(share, 1e-6))                     # weight by the excess of repeated windows over chance
            Dm[i, j] = Dm[j, i] = float(np.median(ratios)) * w if ratios else 0.0
            print(f"pair {real[i]:+d}/{real[j]:+d}: repeated windows {share:.3f} (chance 0.01), echo strength {Dm[i, j]:.3f} from {len(ratios)} windows", flush=True)
    A = Dm; d = float(np.mean([Dm[i, i + 1] for i in range(K - 1)]))
if os.environ.get("D_MODE", "intensity") == "intensity":
    # LOCAL echo maps: each pair (i,j) echoes only in its repeated windows (feathered), zero elsewhere
    Pm = np.zeros((K, K, H, W), np.float16)
    for (i, j), lst in LOCAL.items():
        mp = np.zeros((H, W), np.float32)
        for y0, x0, rr in lst: mp[y0:y0 + 320, x0:x0 + 320] = np.maximum(mp[y0:y0 + 320, x0:x0 + 320], rr)
        mp = ndimage.gaussian_filter(mp, 60); Pm[i, j] = Pm[j, i] = mp
    act = (Pm.astype(np.float32).sum((0, 1)) > 0.02) & v
    print("pixels with a local echo:", round(float(act.mean()), 3), flush=True)
    x = Ye.reshape(K, -1).copy(); idx = np.flatnonzero(act.ravel())
    for c0 in range(0, len(idx), 400000):
        ii = idx[c0:c0 + 400000]; Ac = Pm.reshape(K, K, -1)[:, :, ii].astype(np.float32).transpose(2, 0, 1)   # (n, K, K)
        Ac[:, np.arange(K), np.arange(K)] = 1.0; yc = Ye.reshape(K, -1)[:, ii].T[:, :, None]; xc = yc.copy()
        Lc = (np.linalg.norm(Ac, axis=(1, 2)) ** 2)[:, None, None] + 1e-6
        for _ in range(200): xc = np.maximum(xc - np.einsum("nji,njk->nik", Ac, np.einsum("nij,njk->nik", Ac, xc) - yc) / Lc, 0)
        x[:, ii] = xc[:, :, 0].T
    A_pix = Pm
else:
    y = Ye.reshape(K, -1); x = y.copy(); L = float(np.linalg.norm(A, 2) ** 2)
    for _ in range(300):                                                      # min ||A x - y||^2, x >= 0
        x = np.maximum(x - (A.T @ (A @ x - y)) / L, 0)
own = x.reshape(K, H, W)
# give back ONLY the ink actually removed from the other sheets, to the sheet it came from (not a scaled copy of every
# sheet's own ink): removed on sheet k = its excess over its own ink; split between sources s in proportion to x_s d^|k-s|
ratio = np.clip(own / np.maximum(Ye, 1e-3), 0, 1)                            # legit share of the INK above background
legit = np.where(v, BG + np.maximum(Y - BG, 0) * ratio + np.minimum(Y - BG, 0), Y)
removed = np.maximum(Y - legit, 0)                                             # what each sheet lost
echo_out = np.zeros_like(Y)
xr = own.reshape(K, -1); rem = removed.reshape(K, -1)
for k in range(K):
    contrib = np.stack([xr[s_] * (A_pix[k, s_].reshape(-1).astype(np.float32) if 'A_pix' in globals() else A[k, s_]) if s_ != k else np.zeros_like(xr[k]) for s_ in range(K)])   # who sent ink to k
    tot = contrib.sum(0) + 1e-6
    for s_ in range(K):
        if s_ != k: echo_out[s_] += (rem[k] * contrib[s_] / tot).reshape(H, W)
restored = np.clip(legit + echo_out, 0, 1)
rep = dict(sheets=real, d=round(d, 3), per_sheet={int(o): dict(removed=round(float((Y[k] - legit[k]).sum() / max(np.maximum(Y[k] - BG[k], 0).sum(), 1e-6)), 3),
          given_back=round(float(echo_out[k].sum() / max(np.maximum(Y[k] - BG[k], 0).sum(), 1e-6)), 3)) for k, o in enumerate(real)})
json.dump(rep, open(out + "_unmix.json", "w"), indent=1); print(rep, flush=True)
np.save(out + "_legit.npy", legit.astype(np.float16)); np.save(out + "_restored.npy", restored.astype(np.float16))
def g(a): im = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)); im.thumbnail((420, 420)); return im
cols = [("observe", Y), ("legitime (encre venue d'ailleurs retiree)", legit), ("+ encre rendue (celle qui avait coule)", restored)]
w, h = g(Y[0]).size; G = Image.new("L", (3 * (w + 10), K * (h + 26) + 30), 255); dr = ImageDraw.Draw(G)
for j, (t, arr) in enumerate(cols): dr.text((j * (w + 10) + 4, 6), t, fill=0)
for k, o in enumerate(real):
    for j, (t, arr) in enumerate(cols):
        G.paste(g(arr[k]), (j * (w + 10), 30 + k * (h + 26) + 20))
    dr.text((4, 30 + k * (h + 26) + 4), f"feuille {o:+d}", fill=0)
G.save(out + "_unmix_grid.png")
