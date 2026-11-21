"""First Letters claim image, fully programmatic (no manual annotation of characters).
Input: a deep-read folder (render of the flattened tifxyz mesh, read by Studio and Argus V3 in both layer orders,
p_studio.npz / p_argus-v3.npz with keys fwd / rev) and the letter side (fwd = stored layer order, rev = reversed).
1 render pixel = 1 voxel = 9.362 um (the mesh is isometrically flattened).
 - picks the 2 x 2 cm square where Studio's letter-side ink most exceeds its other-side (control) ink,
 - CLAIM image = Studio letter-side probability (grey, 0..1 linear), with a margin holding line numbers, a 1 cm scale bar
   below, and thin boxes + a size table for 3 letter-sized marks (sizes in px and mm, measured automatically),
 - same square: Studio control (reversed side), Argus V3 letter side, Argus V3 control; whole surface with the square.
usage: make_claim.py DEEP_DIR SIDE NAME OUT_DIR"""
import os, sys, json, numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage, signal
D, SIDE, NAME, OUT = sys.argv[1:5]; os.makedirs(OUT, exist_ok=True)
UM = 9.362; S = int(round(20000 / UM)); OTHER = "rev" if SIDE == "fwd" else "fwd"
def P(m, s): return np.load(f"{D}/p_{m}.npz")[s].astype(np.float32)
st, sc, ar, ac = P("studio", SIDE), P("studio", OTHER), P("argus-v3", SIDE), P("argus-v3", OTHER)
sv = np.load(f"{D}/sv.npy", mmap_mode="r"); valid = np.asarray(sv[sv.shape[0] // 2]) > 0
H, W = valid.shape
try: F = ImageFont.truetype("arial.ttf", 34); Fs = ImageFont.truetype("arial.ttf", 26)
except Exception: F = Fs = ImageFont.load_default()
# 1) square
it = lambda a: np.pad(a, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
I_d, I_v = it((st - sc) * valid), it(valid.astype(np.float64))
def box(I, y, x): return I[y + S, x + S] - I[y, x + S] - I[y + S, x] + I[y, x]
best = (-9, 0, 0)
for y in range(0, max(1, H - S + 1), 8):
    for x in range(0, max(1, W - S + 1), 8):
        cov = box(I_v, y, x) / S ** 2
        if cov < 0.6: continue
        v = box(I_d, y, x) / max(box(I_v, y, x), 1)
        if v > best[0]: best = (v, y, x)
_, y0, x0 = best; S2 = min(S, H - y0, W - x0)
crop = lambda a: a[y0:y0 + S2, x0:x0 + S2]
g = lambda a: (np.clip(a, 0, 1) * 255).astype(np.uint8)
# 2) text lines: peaks of the row profile of letter-side ink
prof = ndimage.gaussian_filter1d((crop(st) > 0.5).mean(1), 25)
pk, _ = signal.find_peaks(prof, distance=int(2500 / UM), prominence=0.05)
# 3) letter-sized marks: connected components 1.5-6 mm tall, compact
lab, n = ndimage.label(ndimage.binary_closing(crop(st) > 0.5, iterations=3))
cands = []
for i, sl in enumerate(ndimage.find_objects(lab), 1):
    h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
    if 2000 / UM <= h <= 4500 / UM and 1500 / UM <= w <= 4500 / UM and 0.6 <= h / w <= 1.8:
        fill = (lab[sl] == i).mean(); cands.append((-(crop(st)[sl] * (lab[sl] == i)).sum(), sl, h, w, fill))
cands.sort(key=lambda c: c[0]); marks = []
for c in cands:
    if all(abs(c[1][0].start - m[1][0].start) + abs(c[1][1].start - m[1][1].start) > 200 for m in marks): marks.append(c)
    if len(marks) == 3: break
# 4) claim image
ML, MB, MT = 140, 260, 70
img = Image.new("RGB", (S2 + ML + 20, S2 + MT + MB), "white"); img.paste(Image.fromarray(g(crop(st))).convert("RGB"), (ML, MT))
d = ImageDraw.Draw(img)
d.text((ML, 12), f"{NAME}  |  Studio ink probability, letter side, 2 x 2 cm = 4 cm2  |  1 px = {UM} um", fill=(0, 0, 0), font=Fs)
for k, p in enumerate(pk, 1):
    d.text((20, MT + p - 18), f"L{k}", fill=(200, 0, 0), font=F); d.line((90, MT + p, ML - 8, MT + p), fill=(200, 0, 0), width=3)
for k, (_, sl, h, w, _) in enumerate(marks, 1):
    d.rectangle((ML + sl[1].start - 6, MT + sl[0].start - 6, ML + sl[1].stop + 6, MT + sl[0].stop + 6), outline=(0, 170, 255), width=2)
    lx = ML + sl[1].stop + 8 if sl[1].stop + 70 < S2 else ML + sl[1].start - 60
    d.text((lx, MT + sl[0].start - 30), f"m{k}", fill=(0, 170, 255), font=Fs)
cm = int(round(10000 / UM)); yb = MT + S2 + 40
d.rectangle((ML, yb, ML + cm, yb + 20), fill=(0, 0, 0)); d.text((ML + cm + 15, yb - 8), "1 cm", fill=(0, 0, 0), font=F)
tx = "  ".join(f"m{k}: {h} x {w} px = {h * UM / 1000:.1f} x {w * UM / 1000:.1f} mm" for k, (_, sl, h, w, _) in enumerate(marks, 1))
d.text((ML, yb + 50), "Letter-sized marks (height x width): " + tx, fill=(0, 0, 0), font=Fs)
d.text((ML, yb + 90), f"Square at render px (row {y0}, col {x0}) of the flattened mesh; lines L1-L{len(pk)} marked in the margin only.", fill=(0, 0, 0), font=Fs)
d.text((ML, yb + 130), "Programmatic: CT volume + mesh -> 16 layers -> Studio (3 depths averaged). No manual edits or annotation of characters.", fill=(0, 0, 0), font=Fs)
img.save(f"{OUT}/{NAME}.png")
# 5) companions
def labelled(a, t):
    im = Image.new("RGB", (S2, S2 + 50), "white"); im.paste(Image.fromarray(g(a)).convert("RGB"), (0, 50)); ImageDraw.Draw(im).text((8, 8), t, fill=(0, 0, 0), font=Fs); return im
tiles = [labelled(crop(st), "Studio - letter side (claim)"), labelled(crop(sc), "Studio - reversed layer order (control)"),
         labelled(crop(ar), "Argus V3 - letter side"), labelled(crop(ac), "Argus V3 - reversed layer order (control)")]
B = Image.new("RGB", (2 * S2 + 20, 2 * (S2 + 50) + 20), "white")
for k, t in enumerate(tiles): B.paste(t, ((k % 2) * (S2 + 20), (k // 2) * (S2 + 70)))
B.resize((B.size[0] // 2, B.size[1] // 2)).save(f"{OUT}/{NAME}_controls.png")
ov = Image.fromarray(g(st)).convert("RGB"); ImageDraw.Draw(ov).rectangle((x0, y0, x0 + S2, y0 + S2), outline=(255, 40, 40), width=10)
ov.resize((W // 2, H // 2)).save(f"{OUT}/{NAME}_whole_surface.png")
inv = lambda a, b: round(float((crop(a)[crop(valid)] > 0.5).mean()), 4)
rep = dict(name=NAME, side=SIDE, square_row_col_size_px=[int(y0), int(x0), int(S2)], lines=len(pk), marks_px=[[int(m[2]), int(m[3])] for m in marks],
           ink_over_0p5_in_square=dict(studio_letter=inv(st, 0), studio_control=inv(sc, 0), argus_letter=inv(ar, 0), argus_control=inv(ac, 0)))
json.dump(rep, open(f"{OUT}/{NAME}_claim.json", "w"), indent=1); print(json.dumps(rep))
