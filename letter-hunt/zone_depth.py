"""Depth re-read of one atlas surface: Argus V3 and Reader V3 read at depth offsets -6..+6 (forward and reversed),
the Interprete letter scan on every map, and per location the number of depths / models where a letter is read.
Real ink letters persist over neighbouring depths and across models; noise flickers.
usage: zone_depth.py SCROLL MESH  -> hunt/depth/<scroll>_<mesh>.npz / .json / _strip.png"""
import csv, glob, io, json, os, shutil, sys, tarfile, types, urllib.request
import numpy as np, torch, zarr
from PIL import Image, ImageDraw, ImageFont
import train as T
from eval_all import load
import interprete as I
import letter_scan as S

MAN = "https://huggingface.co/datasets/rodriguescarson/eligible-scroll-atlas-renders/resolve/main/manifest.csv"
OFFS = (-6, -4, -2, 0, 2, 4, 6)
scroll, mesh = sys.argv[1], sys.argv[2]
os.makedirs("hunt/depth", exist_ok=True); os.makedirs("data/hunt_vols", exist_ok=True)
vp = f"data/hunt_vols/{scroll}_{mesh}.npy"
if not os.path.exists(vp):
    rows = list(csv.DictReader(io.StringIO(urllib.request.urlopen(MAN).read().decode())))
    row = [r for r in rows if r["scroll"] == scroll and r["mesh"] == mesh][0]
    tmp = "hunt/_mesh"; shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
    with urllib.request.urlopen(row["surface_volumes_tar"], timeout=900) as f:
        tarfile.open(fileobj=f, mode="r|").extractall(tmp)
    np.save(vp, np.asarray(zarr.open(glob.glob(f"{tmp}/**/*.zarr", recursive=True)[0], mode="r")["0"])); shutil.rmtree(tmp, ignore_errors=True)
vol = np.load(vp); D = vol.shape[0]; mid = D // 2; valid = vol[mid] > 0
dev = torch.device("cuda"); args = types.SimpleNamespace(patch=256, depth=17, batch=8, amp="bf16", tta=4)
MODELS = os.environ.get("ZD_MODELS", "v3=release/argus-v3/argus-v3.pth,rv3=official/reader-v3.pth")
models = {kv.split("=")[0]: load(kv.split("=")[1])[0].to(dev).eval() for kv in MODELS.split(",")}
TAG = os.environ.get("ZD_TAG", "")


class Seg(T.Segment):
    def __init__(s, v):
        s.vol, s.D = v, v.shape[0]; s.H, s.W = v.shape[1:]; s.sup = v[s.D // 2] > 0; s.ink = s.train_mask = np.zeros_like(s.sup)


it = I.Interprete("runs/interprete_v1.pt")
maps, hits = {}, {}
for n, m in models.items():
    for o in OFFS:
        c = (D - 17) // 2 + o
        if c < 0 or c + 17 > D:
            continue
        f = T.predict(m, Seg(vol[c:c + 17]), args, dev, region=valid)
        r = T.predict(m, Seg(vol[::-1][c:c + 17].copy()), args, dev, region=valid)
        maps[f"{n}_{o:+d}"] = (f * 255).astype(np.uint8); maps[f"{n}_{o:+d}_rev"] = (r * 255).astype(np.uint8)
        fs = np.clip((f - 0.25) / 0.5, 0, 1); target = (fs[valid] > 0.5).mean(); lo = 0.25
        for t in np.linspace(0.25, 0.0, 60):                         # control stretched to the same amount of ink
            if (np.clip((r - t) / 0.5, 0, 1)[valid] > 0.5).mean() >= target:
                lo = t; break
        res = S.scan(fs, np.clip((r - lo) / 0.5, 0, 1), valid, 107.0, it)
        hits[f"{n}_{o:+d}"] = res
        print(n, o, "fwd", len(res["fwd"]), "ctl", len(res["ctl"]), flush=True)
np.savez_compressed(f"hunt/depth/{scroll}_{mesh}{TAG}.npz", ct=vol[mid], **maps)
# persistence: a forward hit counts at a place if the same area (IoU>0.3) has a hit in another depth / model
keys = list(hits)
allf = [(k, h) for k in keys for h in hits[k]["fwd"]]
pers = []
for k, h in allf:
    support = {k2 for k2, h2 in allf if k2 != k and S._iou(h[:3], h2[:3]) > 0.3}
    same = sum(1 for k2, h2 in allf if k2 != k and S._iou(h[:3], h2[:3]) > 0.3 and h2[3] == h[3])
    pers.append(dict(map=k, y=h[0], x=h[1], s=h[2], letter=h[3], p=round(h[4], 2), support=len(support), same_letter=same))
pers = S.nms([(p["y"], p["x"], p["s"], p["letter"], p["support"] + p["p"]) for p in sorted(pers, key=lambda p: -p["support"])])
summary = dict(scroll=scroll, mesh=mesh, per_map={k: (len(v["fwd"]), len(v["ctl"])) for k, v in hits.items()},
               persistent=[dict(y=y, x=x, s=s, letter=c, support=round(sc, 2)) for y, x, s, c, sc in pers if sc >= 5], persistent_ctl=len([1 for k in keys for h in hits[k]["ctl"]]))
json.dump(summary, open(f"hunt/depth/{scroll}_{mesh}{TAG}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("persistent (>=5 other maps agree):", len(summary["persistent"]), [p["letter"] for p in summary["persistent"][:30]])
# picture: v3 at each depth, persistent boxes on the middle one
f = ImageFont.truetype("arial.ttf", 40)
rowsI = []
for o in OFFS:
    k = f"{list(models)[0]}_{o:+d}"
    if k in maps:
        im = Image.fromarray((np.clip((maps[k] / 255.0 - 0.25) / 0.5, 0, 1) * 255).astype(np.uint8)).convert("RGB"); d = ImageDraw.Draw(im); d.text((10, 10), k, fill=(255, 255, 0), font=f)
        if o == 0:
            for p in summary["persistent"]:
                d.rectangle([p["x"], p["y"], p["x"] + p["s"], p["y"] + p["s"]], outline=(255, 60, 60), width=4); d.text((p["x"] + 5, p["y"] + 3), p["letter"], fill=(255, 60, 60), font=f)
        rowsI.append(im)
g = Image.new("RGB", (rowsI[0].width, sum(r.height + 10 for r in rowsI)), "white"); y = 0
for r in rowsI:
    g.paste(r, (0, y)); y += r.height + 10
g.thumbnail((2400, 6000)); g.save(f"hunt/depth/{scroll}_{mesh}{TAG}_strip.png")
