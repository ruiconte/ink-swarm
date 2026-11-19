"""Step 7a (Ruico): align the sheets of a core with ANCHORS = places where a motif repeats.
For each pair of consecutive real sheets: per 3 mm window (step 0.75 mm), search the shift (<= 1.5 mm) that best
matches letter-scale shapes (FFT NCC); anchor if NCC beats the chance level (99th pct of a 6 mm-away null) AND its
neighbouring windows agree (within 0.25 mm). Anchors that continue over >= 2 consecutive pairs (motif on >= 3 sheets)
get double weight. Smooth displacement field = Gaussian-weighted average of anchors (sigma 3 mm); reliability = anchor
density and agreement; unreliable zones (no anchor within 4 mm, or contradictory anchors) = folds/breaks -> mask.
Each sheet is then warped onto the page sheet (displacements chained), iterate 2 times.
usage: align_stack.py CORE_DIR OUT_DIR   (writes aligned sheet maps + mask, same names, for stack_unmix.py)"""
import sys, os, json, numpy as np
from scipy import ndimage
D, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True)
S = json.load(open(os.path.join(D, "summary.json"))); by = {s["offset"]: s for s in S}
offs = sorted(by); real = [offs[0]]
for o in offs[1:]:
    if o - real[-1] >= 12: real.append(o)
Y = {o: np.load(os.path.join(D, f"sheet{o:+d}_{by[o]['reading']}.npy")).astype(np.float32) for o in real}
H, W = Y[real[0]].shape; v = np.all([Y[o] != 0 for o in real], 0)
bp = lambda a: ndimage.gaussian_filter(a, 6) - ndimage.gaussian_filter(a, 40)
Wn, St, M = 320, 80, 160
def best_shift(a, big):
    a = (a - a.mean()) / (a.std() + 1e-6); n = Wn * Wn
    c = np.fft.irfft2(np.fft.rfft2(big) * np.fft.rfft2(a[::-1, ::-1], s=big.shape), s=big.shape)[Wn - 1:Wn - 1 + 2 * M + 1, Wn - 1:Wn - 1 + 2 * M + 1]
    s1 = ndimage.uniform_filter(big, Wn, mode="constant")[Wn // 2:Wn // 2 + 2 * M + 1, Wn // 2:Wn // 2 + 2 * M + 1] * n
    s2 = ndimage.uniform_filter(big ** 2, Wn, mode="constant")[Wn // 2:Wn // 2 + 2 * M + 1, Wn // 2:Wn // 2 + 2 * M + 1] * n
    ncc = c / np.sqrt(np.maximum(s2 - s1 ** 2 / n, 1e-6)) / np.sqrt(n); k = np.unravel_index(np.argmax(ncc), ncc.shape)
    return float(ncc[k]), np.array([k[0] - M, k[1] - M], np.float32)
def anchors(A, B):
    Ab, Bb = bp(A), bp(B); res, null = [], []; rng = np.random.default_rng(0)
    for y in range(M, H - Wn - M, St):
        for x in range(M, W - Wn - M, St):
            if v[y - M:y + Wn + M, x - M:x + Wn + M].mean() < 0.95 or (A[y:y + Wn, x:x + Wn] > 0.5).mean() < 0.02: continue
            r, sh = best_shift(Ab[y:y + Wn, x:x + Wn], Bb[y - M:y + Wn + M, x - M:x + Wn + M]); res.append((y + Wn // 2, x + Wn // 2, r, sh))
            yy, xx = (y + 640) % (H - Wn - 2 * M) + M, (x + 640) % (W - Wn - 2 * M) + M
            null.append(best_shift(Ab[y:y + Wn, x:x + Wn], Bb[yy - M:yy + Wn + M, xx - M:xx + Wn + M])[0])
    if len(null) < 5: return []
    thr = max(0.45, float(np.percentile(null, 99))); hits = [h for h in res if h[2] > thr]
    good = []
    for h in hits:                                                    # neighbouring windows must agree (within 0.25 mm)
        nb = [g for g in hits if g is not h and abs(g[0] - h[0]) <= 2 * St and abs(g[1] - h[1]) <= 2 * St]
        if any(np.abs(g[3] - h[3]).max() <= 27 for g in nb): good.append(h)
    return good
def field(anc, extra=None):
    """smooth displacement field + reliability from anchors (y, x, score, shift, weight)"""
    num = np.zeros((2, H, W), np.float32); den = np.zeros((H, W), np.float32); dis = np.zeros((H, W), np.float32)
    for (y, x, r, sh, w) in anc:
        imp = np.zeros((H, W), np.float32); imp[int(y), int(x)] = w * r
        den += imp; num[0] += imp * sh[0]; num[1] += imp * sh[1]
    s = 280; den_s = ndimage.gaussian_filter(den, s) + 1e-9
    f = np.stack([ndimage.gaussian_filter(num[i], s) / den_s for i in range(2)])
    for (y, x, r, sh, w) in anc:                                     # disagreement of each anchor with the smooth field
        imp = np.zeros((H, W), np.float32); imp[int(y), int(x)] = w * float(np.abs(sh - f[:, int(y), int(x)]).max()); dis += imp
    dis_s = ndimage.gaussian_filter(dis, s) / den_s
    near = ndimage.distance_transform_edt(den == 0) < 430            # an anchor within 4 mm
    reliable = near & (dis_s < 27) & v                               # contradictory (> 0.25 mm) or far = fold / break
    return f, reliable
def warp(a, f):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    return ndimage.map_coordinates(a, [yy + f[0], xx + f[1]], order=1, mode="constant")
p0 = min(real, key=abs); aligned = {p0: Y[p0]}; rel = {p0: v.copy()}; report = {}
order_up = [o for o in real if o > p0]; order_dn = [o for o in real if o < p0][::-1]
for chain in (order_up, order_dn):
    prev = p0; cum = np.zeros((2, H, W), np.float32); cum_rel = v.copy()
    for o in chain:
        cur = Y[o]
        for it in range(2):                                          # iterate: align, find more anchors, refine
            anc = anchors(aligned[prev], warp(cur, cum) if it == 0 else warp(cur, cum + f_it))
            if it == 0 and not anc: f_it = np.zeros((2, H, W), np.float32); rel_it = np.zeros((H, W), bool); break
            # motif repeated on >= 3 sheets: anchor also matches the next-next sheet -> double weight
            w_anc = [(y, x, r, sh, 2.0 if (prev != p0) else 1.0) for (y, x, r, sh) in anc]
            f_new, rel_it = field(w_anc); f_it = f_new if it == 0 else f_it + f_new
        cum = cum + f_it; cum_rel = cum_rel & rel_it
        aligned[o] = warp(cur, cum); rel[o] = cum_rel.copy()
        report[o] = dict(anchors=len(anc) if anc else 0, reliable_share=round(float(cum_rel[v].mean()), 3),
                         median_shift_mm=[round(float(np.median(cum[i][v]) * 0.009362), 3) for i in range(2)])
        print(f"sheet {o:+d}: {report[o]}", flush=True); prev = o
for o in real:
    np.save(os.path.join(OUT, f"sheet{o:+d}_{by[o]['reading']}.npy"), np.where(rel[o], aligned[o], 0).astype(np.float16))
json.dump([by[o] for o in real], open(os.path.join(OUT, "summary.json"), "w")); json.dump(report, open(os.path.join(OUT, "align_report.json"), "w"), indent=1)
