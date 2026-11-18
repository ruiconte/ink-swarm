"""Rank HerculeX pieces by HerculeX's own per-piece side check: ink on the written face ("near") against ink with the
depth layers scrambled ("scrambled"). Real writing gives a lot of ink on the written face and almost none once the
layers are scrambled; noise and fibres survive the scrambling. Writes a ranked JSON and a contact sheet of the top pieces.
usage: side_rank.py FACES_ROOT OUT_DIR [top_n]"""
import glob, json, os, sys
import numpy as np
from PIL import Image, ImageDraw

root, out = sys.argv[1], sys.argv[2]; n = int(sys.argv[3]) if len(sys.argv) > 3 else 20
os.makedirs(out, exist_ok=True); rows = []
for f in glob.glob(f"{root}/**/*-taught-side.json", recursive=True):
    d = json.load(open(f)); near, scr = d.get("near", 0), d.get("scrambled", 0)
    if near > 0.08:
        rows.append(dict(score=round(near / (scr + 0.02), 2), near=round(near, 3), scrambled=round(scr, 3), piece=os.path.relpath(f, root)))
rows.sort(key=lambda r: -r["score"])
json.dump(rows, open(f"{out}/side_rank.json", "w"), indent=1)
tiles = []
for r in rows[:n]:
    a = np.asarray(Image.open(os.path.join(root, r["piece"]).replace("-side.json", ".png")), np.float32)
    g = (a - np.percentile(a, 1)) / (np.percentile(a, 99.5) - np.percentile(a, 1) + 1e-6)
    im = Image.fromarray((np.clip(g, 0, 1) * 215 + 40).astype(np.uint8)); im.thumbnail((360, 360))
    t = Image.new("L", (360, 380), 255); t.paste(im, (0, 20)); ImageDraw.Draw(t).text((2, 2), f"{r['near']:.2f}/{r['scrambled']:.2f}", fill=0); tiles.append(t)
if tiles:
    g = Image.new("L", (5 * 365, -(-len(tiles) // 5) * 385), 255)
    for k, t in enumerate(tiles):
        g.paste(t, ((k % 5) * 365, (k // 5) * 385))
    g.save(f"{out}/side_top.png")
print(len(rows), "pieces ranked")
