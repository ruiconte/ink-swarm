"""Ruico: sheets may have SLIPPED, so a transferred letter sits a bit further on the neighbour. For each 3 mm window
of the page, search the neighbour over shifts up to +-1.5 mm (FFT cross-correlation of letter-scale band-passed ink),
keep the best shift and its score. Anti-chance tests:
(1) null = same search against a far window of the neighbour (no real link possible),
(2) COHERENCE: a real slip moves the sheet as a block, so neighbouring windows must agree on the shift.
Also (3) density exchange: is the page empty where the neighbour is full (ink migrated) -> correlation of 3 mm ink
density maps between sheets (negative = exchange). usage: slip.py CORE_DIR PAGE.npy NEIGHBOUR.npy [...]"""
import sys, os, json, numpy as np
from scipy import ndimage
D = sys.argv[1]; B = np.load(os.path.join(D, sys.argv[2])).astype(np.float32); v = B != 0
W, S, M = 320, 160, 160                                     # window 3 mm, step 1.5 mm, max shift 1.5 mm
bp = lambda a: ndimage.gaussian_filter(a, 6) - ndimage.gaussian_filter(a, 40)
Bb = bp(B); rng = np.random.default_rng(0); rep = {}
def best_shift(a, big):
    """a: W x W window, big: (W+2M) x (W+2M) region of the neighbour -> best NCC and shift"""
    a = (a - a.mean()) / (a.std() + 1e-6); F = np.fft.rfft2(big, s=big.shape); K = np.fft.rfft2(a[::-1, ::-1], s=big.shape)
    c = np.fft.irfft2(F * K, s=big.shape)[W - 1:W - 1 + 2 * M + 1, W - 1:W - 1 + 2 * M + 1]
    win_sum = ndimage.uniform_filter(big, W, mode="constant")[W // 2:W // 2 + 2 * M + 1, W // 2:W // 2 + 2 * M + 1] * W * W
    win_sq = ndimage.uniform_filter(big ** 2, W, mode="constant")[W // 2:W // 2 + 2 * M + 1, W // 2:W // 2 + 2 * M + 1] * W * W
    sd = np.sqrt(np.maximum(win_sq - win_sum ** 2 / (W * W), 1e-6)); ncc = c / sd
    k = np.unravel_index(np.argmax(ncc), ncc.shape); return float(ncc[k] / (W * W) ** 0.5), (k[0] - M, k[1] - M)
dens = lambda a: ndimage.gaussian_filter(((a > 0.5) & (a != 0)).astype(np.float32), 160)
for nf in sys.argv[3:]:
    X = np.load(os.path.join(D, nf)).astype(np.float32); Xb = bp(X); vx = v & (X != 0)
    res, null = [], []
    for y in range(M, B.shape[0] - W - M, S):
        for x in range(M, B.shape[1] - W - M, S):
            if vx[y - M:y + W + M, x - M:x + W + M].mean() < 0.95 or (B[y:y + W, x:x + W] > 0.5).mean() < 0.03: continue
            r, sh = best_shift(Bb[y:y + W, x:x + W], Xb[y - M:y + W + M, x - M:x + W + M]); res.append((y, x, r, sh))
            yy, xx = rng.integers(M, B.shape[0] - W - M), rng.integers(M, B.shape[1] - W - M)       # far window: no real link
            if vx[yy - M:yy + W + M, xx - M:xx + W + M].mean() >= 0.95: null.append(best_shift(Bb[y:y + W, x:x + W], Xb[yy - M:yy + W + M, xx - M:xx + W + M])[0])
    if len(res) < 5 or len(null) < 5: rep[nf] = "too few windows"; print(nf, rep[nf]); continue
    thr = float(np.percentile(null, 99)); hits = [h for h in res if h[2] > thr]
    # coherence: for each hit, do its grid neighbours (also hits) agree on the shift within 0.25 mm?
    pos = {(h[0], h[1]): h[3] for h in hits}; agree = 0
    for (y, x), sh in pos.items():
        nb = [pos[k] for k in ((y + S, x), (y - S, x), (y, x + S), (y, x - S)) if k in pos]
        if any(abs(sh[0] - s2[0]) <= 27 and abs(sh[1] - s2[1]) <= 27 for s2 in nb): agree += 1
    m = vx & (ndimage.binary_erosion(vx, iterations=80))
    dc = float(np.corrcoef(dens(B)[m], dens(X)[m])[0, 1]) if m.sum() > 1000 else None
    rep[nf] = dict(windows=len(res), null_99=round(thr, 3), shifted_matches=len(hits), share=round(len(hits) / len(res), 3),
                   coherent_matches=agree, median_shift_px=[int(np.median([h[3][0] for h in hits])), int(np.median([h[3][1] for h in hits]))] if hits else None,
                   density_corr=round(dc, 3) if dc is not None else None)
    print(nf, rep[nf], flush=True)
json.dump(rep, open(os.path.join(D, "slip_report.json"), "w"), indent=1)
