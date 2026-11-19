"""Ink legitimacy, pixel level (Ruico's stack rule, corrected for chance). For every ink pixel of page sheet N:
- presence on each real neighbouring sheet (same place within 6 px), chain of consecutive presences outwards,
- per neighbour o: observed shared rate q_o and chance rate c_o (neighbour shifted 6 mm); the evidence that a shared
  pixel is a real transfer is e_o = 1 - c_o / q_o (0 if no excess over chance),
- P(not N's ink) = 1 - prod(1 - e_o) over the chain, applied only if the chain's PEAK (trend) is not on N,
- the page loses P x ink at those pixels, the peak sheet gets it back (mass conserved); flat chains (>= 3 sheets,
  similar strength) are lowered on N and given to nobody.
usage: attrib3.py CORE_DIR OUT_PREFIX"""
import sys, os, json, numpy as np
from scipy import ndimage
from PIL import Image, ImageDraw
D, out = sys.argv[1], sys.argv[2]
S = json.load(open(os.path.join(D, "summary.json"))); by = {s["offset"]: s for s in S}
def load(off): s = by[off]; return np.load(os.path.join(D, f"sheet{off:+d}_{s['reading']}.npy")).astype(np.float32)
offs = sorted(by); p0 = min(offs, key=abs); real = [p0]
for o in sorted(o for o in offs if o > p0):
    if o - real[-1] >= 12: real.append(o)
for o in sorted((o for o in offs if o < p0), reverse=True):
    if min(real) - o >= 12: real.insert(0, o)
st = {o: load(o) for o in real}; i0 = real.index(p0); B = st[p0]; v = B != 0; bi = (B > 0.5) & v
SL = json.load(open(os.path.join(D, "slip_report.json"))) if os.path.exists(os.path.join(D, "slip_report.json")) else {}
for o in real:                                             # Ruico: sheets slipped -> compare with the neighbour moved back by its coherent slip
    r = SL.get(f"sheet{o:+d}_{by[o]['reading']}.npy")
    if o != p0 and isinstance(r, dict) and r.get("coherent_matches", 0) >= 3 and r.get("median_shift_px"):
        dy, dx = r["median_shift_px"]; st[o] = np.roll(np.roll(st[o], -dy, 0), -dx, 1); print(f"sheet {o:+d}: slip corrected by {dy},{dx} px", flush=True)
dil = {o: ndimage.maximum_filter(st[o], 13) for o in real}
ev = {}
for o in real:
    if o == p0: continue
    q = float(((dil[o] > 0.5) & bi).sum() / bi.sum()); c = float(((np.roll(np.roll(dil[o], 640, 0), 640, 1) > 0.5) & bi).sum() / bi.sum())
    ev[o] = max(0.0, 1 - c / q) if q > 0 else 0.0
    print(f"sheet {o:+d}: shared {q:.3f} chance {c:.3f} -> evidence per shared pixel {ev[o]:.2f}", flush=True)
H, W = B.shape; prof = np.stack([np.where(bi, dil[o], 0) for o in real])               # (sheets, H, W) strength at page ink
pres = prof > 0.5
up = np.zeros((H, W), np.int8); dn = np.zeros((H, W), np.int8); alive = bi.copy()
for k in range(i0 + 1, len(real)): alive &= pres[k]; up += alive
alive = bi.copy()
for k in range(i0 - 1, -1, -1): alive &= pres[k]; dn += alive
keep_none = 1.0; P = np.zeros((H, W), np.float32); notown = np.ones((H, W), np.float32)
for j in range(1, len(real) - i0):                                                      # chain evidence upwards
    o = real[i0 + j]; notown *= np.where(up >= j, 1 - ev[o], 1)
for j in range(1, i0 + 1):
    o = real[i0 - j]; notown *= np.where(dn >= j, 1 - ev[o], 1)
P = (1 - notown) * bi
if os.environ.get("SOURCE", "strong") == "sharp":
    # the original is SHARPER than its copy (a copy is blurred by soaking): local edge energy of each sheet's ink
    shp = np.stack([np.where(bi, ndimage.gaussian_filter(np.abs(ndimage.laplace(ndimage.gaussian_filter(st[o], 2))), 8), 0) for o in real])
    peak = np.argmax(np.where(pres | (np.arange(len(real))[:, None, None] == i0), shp, -1), 0)
else:
    peak = np.argmax(np.where(pres | (np.arange(len(real))[:, None, None] == i0), prof, -1), 0)   # sheet where the chain is strongest
chain = up + dn; seg_max = prof.max(0); seg_min = np.where(pres, prof, 9).min(0)
flat = (chain >= 2) & (np.minimum(seg_min, B) >= 0.8 * seg_max)
received = bi & (chain > 0) & (peak != i0) & ((prof[peak, np.arange(H)[:, None], np.arange(W)[None]] > B * 1.1) | (os.environ.get('SOURCE') == 'sharp')) & ~flat
P = np.where(received | flat, P, 0)
# soft and homogeneous (Ruico): the removed ink goes to the LOCAL BACKGROUND level, not to black, with feathered edges
SPREAD = int(os.environ.get("SPREAD", "24"))               # Ruico: wider, blurrier zones, at the scale of the image blur (strokes 30-50 px)
Ps = np.clip(ndimage.gaussian_filter(ndimage.maximum_filter(P, 2 * SPREAD + 1), SPREAD), 0, 1) * v
nonink = v & (B < 0.5); bg = ndimage.gaussian_filter(np.where(nonink, B, 0), 40) / np.maximum(ndimage.gaussian_filter(nonink.astype(np.float32), 40), 1e-3)
bg = np.minimum(bg, B)
moved = Ps * (B - bg); clean = B - moved; gains = {}
# give the moved ink back to the sheet it came from: each pixel of the soft zone takes the source of the nearest received pixel
srcmap = np.where(received, peak, -1)
if received.any():
    _, (iy, ix) = ndimage.distance_transform_edt(~received, return_indices=True); srcmap = peak[iy, ix]
    srcmap = np.where(ndimage.distance_transform_edt(~received) <= 3 * SPREAD, srcmap, -1)
for k, o in enumerate(real):
    if o == p0: continue
    g = ndimage.gaussian_filter(np.where(srcmap == k, moved, 0) * (Ps > 0.02), SPREAD / 2) * v   # smooth: no sharp seams
    if g.any():
        gains[o] = float(g.sum()); np.save(f"{out}_sheet{o:+d}_restored.npy", np.clip(st[o] + g, 0, 1).astype(np.float16))
np.save(out + "_page_clean3.npy", clean.astype(np.float16)); np.save(out + "_P3.npy", P.astype(np.float16))
rep = dict(sheets=real, page=p0, evidence=ev, received_px=round(float(received.sum() / bi.sum()), 3), flat_px=round(float(flat.sum() / bi.sum()), 3),
           ink_mass_moved=round(float(moved.sum() / (B * bi).sum()), 3), mean_P_where_moved=round(float(P[P > 0].mean()) if (P > 0).any() else 0, 3), given_back=gains)
json.dump(rep, open(out + "_stack3.json", "w"), indent=1); print(rep, flush=True)
def rgb(a, heat=None):
    g = (np.clip(a, 0, 1) * 255).astype(np.uint8); im = np.stack([g] * 3, -1).astype(np.float32)
    if heat is not None: im = im * (1 - heat[..., None]) + heat[..., None] * np.array([255, 0, 0])
    return Image.fromarray(im.clip(0, 255).astype(np.uint8))
n20 = int(20 / 0.009362); ys, xs = np.nonzero(v); cy = int(np.median(ys)); best = None
for x0 in range(0, W - n20, n20 // 2):
    y0 = max(0, min(H - n20, cy - n20 // 2)); sc = bi[y0:y0 + n20, x0:x0 + n20].mean()
    if best is None or sc > best[0]: best = (sc, y0, x0)
_, y0, x0 = best; sl = (slice(y0, y0 + n20), slice(x0, x0 + n20))
crops = [rgb(B[sl]), rgb(B[sl], Ps[sl]), rgb(clean[sl])]
G = Image.new("RGB", (3 * 820, 860), "white"); d = ImageDraw.Draw(G)
for i, (c, t) in enumerate(zip(crops, ("page avant", "rouge = probabilite que l'encre vienne d'une autre feuille", "page apres deplacement proportionnel"))):
    G.paste(c.resize((800, 800)), (i * 820, 50)); d.text((i * 820 + 4, 20), t, fill=(0, 0, 0))
G.save(f"{out}_stack3_crop20mm.jpg", quality=90)
for name, im in (("stack3_before", rgb(B)), ("stack3_P", rgb(B, Ps)), ("stack3_after", rgb(clean))):
    im.thumbnail((1400, 1400)); im.save(f"{out}_{name}.jpg", quality=90)
