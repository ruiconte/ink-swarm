"""Paste the ink maps of one HerculeX mosaic (layout group) onto the unrolled sheet, using layout.json positions.
Argus V3 maps first, reader-v3 maps where Argus has none. Grey display (1st..99.5th percentile).
usage: mosaic.py LAYOUT_JSON GROUP_OR_PIECE OUT_PNG ROOT [ROOT ...]   (GROUP_OR_PIECE: g136 or unit/piece e.g. rows-0033-0045/6000041)"""
import glob, json, os, sys
import numpy as np
from PIL import Image

lay, sel, out, roots = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
P = json.load(open(lay))["pieces"]
if sel.startswith("g"):
    g = int(sel[1:])
else:
    u, pc = sel.split("/"); g = [p["group"] for p in P if p["unit"] == u and str(p["piece"]) == pc][0]
ps = [p for p in P if p["group"] == g]
PX = float(os.environ.get("MOSAIC_PX", "0.04"))                                                    # mm per output pixel (40 um)
x0 = min(p["x_mm"] for p in ps); y0 = min(p["y_mm"] for p in ps)
W = int((max(p["x_mm"] + p["width_mm"] for p in ps) - x0) / PX) + 1
H = int((max(p["y_mm"] + p["height_mm"] for p in ps) - y0) / PX) + 1
acc = np.zeros((H, W), np.float32); cnt = np.zeros((H, W), np.float32); used = []
for p in ps:
    f = None
    for r in roots:
        c = glob.glob(f"{r}/**/{p['unit']}/piece-{p['piece']:04d}-taught.png", recursive=True)
        if c:
            f = c[0]; break
    if not f:
        continue
    a = np.asarray(Image.open(f), np.float32) / 255
    if p.get("flip_rows"): a = a[::-1]
    if p.get("flip_cols"): a = a[:, ::-1]
    w, h = int(p["width_mm"] / PX), int(p["height_mm"] / PX)
    a = np.asarray(Image.fromarray(a).resize((w, h), Image.BILINEAR))
    X, Y = int((p["x_mm"] - x0) / PX), int((p["y_mm"] - y0) / PX)
    w, h = min(w, W - X), min(h, H - Y)
    m = a[:h, :w] > 0.005
    free = cnt[Y:Y + h, X:X + w] == 0; mm = m & free if os.environ.get('MOSAIC_FIRST') else m
    acc[Y:Y + h, X:X + w] += a[:h, :w] * mm; cnt[Y:Y + h, X:X + w] += mm; used.append((p["unit"], p["piece"], os.path.basename(os.path.dirname(os.path.dirname(f)))))
img = acc / np.maximum(cnt, 1)
v = img[cnt > 0]
lo, hi = (np.percentile(v, 1), np.percentile(v, 99.5)) if v.size else (0, 1)
g8 = (np.clip((img - lo) / (hi - lo + 1e-6), 0, 1) * 215 + 40).astype(np.uint8); g8[cnt == 0] = 0
Image.fromarray(g8).save(out)
print("group", g, "pieces", len(ps), "pasted", len(used), "size mm", round(W * PX, 1), "x", round(H * PX, 1))
