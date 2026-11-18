"""Letter scan over HerculeX piece ink maps (*-taught.png). Control = the same map transposed (identical ink amount and
texture, but letter shapes no longer upright), so only orientation-dependent letter structure scores above control.
usage: hx_letterscan.py FACES_ROOT OUT_JSON [px_mm]"""
import glob, json, os, sys
import numpy as np
from PIL import Image
from scipy import ndimage
import interprete as I
import letter_scan as S

root, out = sys.argv[1], sys.argv[2]
PX = float(sys.argv[3]) if len(sys.argv) > 3 else 1000 / 7.91
DS = 4                                                   # work at 1/4 resolution (crops are resized to 64 px anyway)
it = I.Interprete("runs/interprete_v1.pt")
done = {}
if os.path.exists(out):
    done = {r["piece"]: r for r in json.load(open(out, encoding="utf-8"))}
fs = sorted(glob.glob(f"{root}/**/*-taught.png", recursive=True))
for k, f in enumerate(fs):
    key = os.path.relpath(f, root)
    if key in done:
        continue
    im = Image.open(f); im = im.resize((max(1, im.width // DS), max(1, im.height // DS)), Image.BILINEAR)
    a = np.asarray(im, np.float32) / 255
    m = np.clip((a - 0.25) / 0.5, 0, 1)
    valid = ndimage.binary_closing(a > 0.02, iterations=8) | (a > 0.02)
    valid = ndimage.binary_dilation(valid, iterations=6)
    cm2 = valid.sum() / (PX / DS * 10) ** 2
    if cm2 < 0.05:
        done[key] = dict(piece=key, cm2=round(float(cm2), 3), fwd=0, ctl=0, hits=[]); continue
    res = S.scan(m, m.T.copy(), valid, PX / DS, it) if m.shape[0] == m.shape[1] else None
    if res is None:                                       # transpose changes shape: scan both with their own valid
        r1 = S.scan(m, m, valid, PX / DS, it); r2 = S.scan(m.T.copy(), m.T.copy(), valid.T.copy(), PX / DS, it)
        res = dict(fwd=r1["fwd"], ctl=r2["fwd"])
    done[key] = dict(piece=key, cm2=round(float(cm2), 3), fwd=len(res["fwd"]), ctl=len(res["ctl"]),
                     hits=[(int(y) * DS, int(x) * DS, int(s) * DS, c, round(p, 2)) for y, x, s, c, p in res["fwd"]])
    if k % 50 == 0:
        json.dump(list(done.values()), open(out, "w", encoding="utf-8"), ensure_ascii=False)
        F = sum(r["fwd"] for r in done.values()); C = sum(r["ctl"] for r in done.values())
        print(k, len(fs), "fwd", F, "ctl", C, flush=True)
json.dump(list(done.values()), open(out, "w", encoding="utf-8"), ensure_ascii=False)
R = sorted(done.values(), key=lambda r: -(r["fwd"] - r["ctl"]) / max(r["cm2"], 0.2))
print("TOTAL fwd", sum(r["fwd"] for r in R), "ctl", sum(r["ctl"] for r in R))
print("TOP", [(r["piece"][-40:], r["fwd"], r["ctl"], r["cm2"]) for r in R[:15]])
