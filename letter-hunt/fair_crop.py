"""Discord picture: Reader V3 vs Argus V3 on PHerc0841 w00 (never seen in training), both rank-normalised (same
brightness/contrast), on the crops with the most labelled letters. -> runs/fair_crop_<n>.png"""
import types, numpy as np, torch
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
import train as T
from eval_all import load

seg = "pherc0841-w00-native"
dev = torch.device("cuda"); args = types.SimpleNamespace(patch=256, depth=17, batch=8, amp="bf16", tta=4)
s = T.Segment(seg); m = s.sup
P = {n: T.predict(load(p)[0].to(dev).eval(), s, args, dev, region=m) for n, p in
     (("Reader V3", "official/reader-v3.pth"), ("Argus V3", "release/argus-v3/argus-v3.pth"))}


def ranknorm(p):
    out = np.zeros_like(p, np.float32); v = p[m]
    out[m] = (np.argsort(np.argsort(v, kind="stable"), kind="stable") + 0.5) / len(v)
    return np.clip((out - 0.55) / 0.4, 0, 1)                 # same display curve for both


R = {n: ranknorm(p) for n, p in P.items()}
f = ImageFont.truetype("arial.ttf", 30)
lab, n = ndimage.label(ndimage.binary_closing(m, iterations=10))
boxes = []
for k, sl in enumerate(ndimage.find_objects(lab), 1):
    ink = s.ink_eval[sl][lab[sl] == k].sum()
    if ink > 20000: boxes.append((ink, sl))
boxes = [b for _, b in sorted(boxes, key=lambda b: -b[0])[:4]]
rows = []
for sl in boxes:
    ys, xs = sl; y0, y1, x0, x1 = ys.start, ys.stop, xs.start, xs.stop
    tiles = [R["Reader V3"][y0:y1, x0:x1], R["Argus V3"][y0:y1, x0:x1], s.ink_eval[y0:y1, x0:x1].astype(np.float32)]
    sc = 420 / (y1 - y0); w2 = int((x1 - x0) * sc)
    rows.append([Image.fromarray((a * 255).astype(np.uint8)).resize((w2, 420), Image.LANCZOS) for a in tiles])
gap = 16; Wd = max(r[0].width for r in rows)
out = Image.new("RGB", (3 * Wd + 2 * gap, 50 + len(rows) * (420 + gap)), "white"); dr = ImageDraw.Draw(out)
for i, tt in enumerate(("Reader V3", "Argus V3", "expert labels")):
    dr.text((i * (Wd + gap) + 8, 10), tt, fill=(0, 0, 0), font=f)
for j, r in enumerate(rows):
    for i, im in enumerate(r):
        out.paste(im, (i * (Wd + gap), 50 + j * (420 + gap)))
out.save("runs/fair_crop.png"); print(len(rows), out.size)
