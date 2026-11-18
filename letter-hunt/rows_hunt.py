"""Find WRITTEN LINES in HerculeX pieces: rows of >= 4 letter-scan hits at the same height (|dy| < 0.4 size),
similar sizes, gaps 0.6..2.2 letter widths. Same test on the transposed map as control.
usage: rows_hunt.py LETTERSCAN_JSON FACES_ROOT OUT_DIR"""
import json, os, sys
import numpy as np
from PIL import Image
import interprete as I
import letter_scan as S

js, root, out = sys.argv[1:4]; os.makedirs(out, exist_ok=True)
PX = 1000 / float(os.environ.get("VOXEL_UM", "9.362")); DS = 4
it = I.Interprete("runs/interprete_v1.pt")


def rows(h):
    h = sorted(h, key=lambda t: t[1]); best = []
    for i, a in enumerate(h):
        chain = [a]
        for b in h[i + 1:]:
            c = chain[-1]; s = (c[2] + b[2]) / 2
            if abs((b[0] + b[2] / 2) - (c[0] + c[2] / 2)) < 0.4 * s and 0.6 * s <= (b[1] - c[1]) <= 2.2 * s and 0.6 < b[2] / c[2] < 1.6:
                chain.append(b)
        if len(chain) > len(best):
            best = chain
    return best


R = json.load(open(js, encoding="utf-8")); found = []
for r in R:
    if r["fwd"] < 4:
        continue
    f = rows([tuple(h) for h in r["hits"]])
    if len(f) < 4:
        continue
    im = Image.open(os.path.join(root, r["piece"])); im = im.resize((im.width // DS, im.height // DS), Image.BILINEAR)
    m = np.clip((np.asarray(im, np.float32) / 255 - 0.25) / 0.5, 0, 1).T.copy(); valid = m > -1
    c = S.scan(m, m, valid, PX / DS, it)["fwd"]
    cr = rows([(y * DS, x * DS, s * DS, l, p) for y, x, s, l, p in c])
    found.append(dict(piece=r["piece"], row=len(f), ctl_row=len(cr), letters="".join(h[3] for h in f), hits=f))
    print(r["piece"][-45:], len(f), "ctl", len(cr), "".join(h[3] for h in f), flush=True)
found.sort(key=lambda d: -(d["row"] - d["ctl_row"]))
json.dump(found, open(f"{out}/rows.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
for k, d in enumerate(found[:12]):                      # grey render (no clipping), crop around the row
    a = np.asarray(Image.open(os.path.join(root, d["piece"])), np.float32)
    ys = [h[0] for h in d["hits"]]; xs = [h[1] for h in d["hits"]]; s = int(np.mean([h[2] for h in d["hits"]]))
    y0, y1 = max(0, min(ys) - 2 * s), min(a.shape[0], max(ys) + 3 * s); x0, x1 = max(0, min(xs) - s), min(a.shape[1], max(xs) + 2 * s)
    g = a[y0:y1, x0:x1]; g = (g - np.percentile(g, 1)) / (np.percentile(g, 99.5) - np.percentile(g, 1) + 1e-6)
    Image.fromarray((np.clip(g, 0, 1) * 200 + 40).astype(np.uint8)).save(f"{out}/row_{k:02d}.png")
print("pieces with a row >=4:", len(found), "| row beats control:", sum(d["row"] > d["ctl_row"] for d in found))
