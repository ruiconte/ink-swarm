"""Calibrate the ink-legitimacy rules on papyrologist-validated letters (PHerc0841 w00): the page sheet of the core is
the labelled sheet, pixel-aligned with the expert ink labels. For each real neighbouring sheet:
- how much of the TRUE ink (labels) is also ink on the neighbour (vs chance, neighbour shifted 6 mm),
- how much of the page's FALSE ink (model ink outside the labels) is explained by the neighbour (offset candidates),
- strength ratio neighbour/page on true letters (how strongly real letters echo on the next sheets),
- the legitimacy program's decision on true vs false ink: what share of TRUE ink it would remove (must be small).
usage: calib_offset.py CORE_DIR LABELS.npz"""
import sys, os, json, numpy as np
from scipy import ndimage
D, L = sys.argv[1], np.load(sys.argv[2])
S = json.load(open(os.path.join(D, "summary.json"))); by = {s["offset"]: s for s in S}
def load(o): return np.load(os.path.join(D, f"sheet{o:+d}_{by[o]['reading']}.npy")).astype(np.float32)
offs = sorted(by); p0 = min(offs, key=abs); real = [p0]
for o in sorted(o for o in offs if o > p0):
    if o - real[-1] >= 12: real.append(o)
for o in sorted((o for o in offs if o < p0), reverse=True):
    if min(real) - o >= 12: real.insert(0, o)
B = load(p0); ink = L["ink"] > 0; sup = L["sup"] > 0
H, W = min(B.shape[0], ink.shape[0]), min(B.shape[1], ink.shape[1]); B, ink, sup = B[:H, :W], ink[:H, :W], sup[:H, :W]
v = (B != 0) & sup; true_ink = ink & v & (B > 0.5); false_ink = ~ink & v & (B > 0.5)
rep = dict(sheets=real, page=p0, page_true_ink_px=int(true_ink.sum()), page_false_ink_px=int(false_ink.sum()), page_auc_proxy=None, neighbours={})
for o in real:
    if o == p0: continue
    X = load(o)[:H, :W]; xd = ndimage.maximum_filter(X, 13); xs = np.roll(np.roll(xd, 640, 0), 640, 1)
    r = dict(true_shared=float(((xd > 0.5) & true_ink).sum() / max(true_ink.sum(), 1)), true_chance=float(((xs > 0.5) & true_ink).sum() / max(true_ink.sum(), 1)),
             false_shared=float(((xd > 0.5) & false_ink).sum() / max(false_ink.sum(), 1)), false_chance=float(((xs > 0.5) & false_ink).sum() / max(false_ink.sum(), 1)),
             echo_strength_on_true=float(np.median(xd[true_ink & (xd > 0.5)] / np.maximum(B[true_ink & (xd > 0.5)], 1e-3))) if (true_ink & (xd > 0.5)).any() else None,
             neighbour_stronger_on_true=float(((xd > B * 1.1) & true_ink).sum() / max(true_ink.sum(), 1)))
    rep["neighbours"][o] = {k: (round(x, 3) if isinstance(x, float) else x) for k, x in r.items()}
    print(o, rep["neighbours"][o], flush=True)
if os.path.exists(os.path.join(D, "C_P3.npy")):                      # the legitimacy program's own decision on this core
    P = np.load(os.path.join(D, "C_P3.npy")).astype(np.float32)[:H, :W]
    rep["program_removes_true_ink"] = round(float(P[true_ink].mean()), 3); rep["program_removes_false_ink"] = round(float(P[false_ink].mean()), 3)
    print("program: mean P on TRUE ink", rep["program_removes_true_ink"], "on FALSE ink", rep["program_removes_false_ink"], flush=True)
json.dump(rep, open(os.path.join(D, "calib_offset.json"), "w"), indent=1)
