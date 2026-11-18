"""Letter proposals on a re-read zone: the forward maps of ALL models and depths are averaged (what everybody sees),
each persistent spot is read by the Interprete (top-3) and shown next to the 5 most similar REAL papyrus letters
(HomerComp, CC BY-NC) in the Interprete's feature space. The same is done on the averaged control maps (reversed
layers) so we can see how many confident proposals noise alone gives.
For our own eyes only: proposals, not claims (prize rule: no manual letter annotation in a claim).
usage: propose.py SCROLL MESH [MESH ...] -> hunt/proposals/<scroll>_<mesh>.png / .json"""
import glob, json, os, sys
import numpy as np, torch
from PIL import Image, ImageDraw, ImageFont
import interprete as I
import letter_scan as S

scroll, meshes = sys.argv[1], sys.argv[2:]
os.makedirs("hunt/proposals", exist_ok=True)
it = I.Interprete("runs/interprete_v1.pt"); net = it.net.eval(); dev = it.dev
feat_net = net.f[:-2]


def feats(X):
    out = []
    with torch.no_grad():
        for k in range(0, len(X), 1024):
            x = torch.as_tensor(np.asarray(X[k:k + 1024], np.float32)).to(dev)[:, None]
            out.append(torch.nn.functional.normalize(feat_net(x).float(), dim=1).cpu())
    return torch.cat(out)


L = np.load("hunt/letters_homercomp.npz")
LF = feats(L["shapes"].astype(np.float32)); LG, LL = L["greys"], L["labels"]
st = lambda a: np.clip((a / 255.0 - 0.25) / 0.5, 0, 1)
font = ImageFont.truetype("arial.ttf", 20); small = ImageFont.truetype("arial.ttf", 15)
for mesh in meshes:
    zs = [np.load(f) for f in sorted(glob.glob(f"hunt/depth/{scroll}_{mesh}*.npz"))]
    fw = [st(z[k]) for z in zs for k in z.files if k != "ct" and not k.endswith("_rev")]
    rv = [st(z[k]) for z in zs for k in z.files if k.endswith("_rev")]
    avg, ctl = np.mean(fw, 0), np.mean(rv, 0); ct = zs[0]["ct"]; valid = ct > 0
    avg, ctl = np.clip((avg - 0.1) / 0.5, 0, 1), np.clip((ctl - 0.1) / 0.5, 0, 1)
    spots = []
    for f in glob.glob(f"hunt/depth/{scroll}_{mesh}*.json"):
        if f.endswith("_pers.json"):
            continue
        spots += [(p["y"], p["x"], p["s"], p["letter"], p["support"]) for p in json.load(open(f, encoding="utf-8"))["persistent"]]
    spots = S.nms(spots)
    res = S.scan(avg, ctl, valid, 107.0, it)                       # how many confident letters noise alone gives
    rows = []
    for y, x, s, _, sup in spots:
        c = S.crop64(avg, y, x, s); p = it.predict(c[None])[0]; top = np.argsort(-p[:-1])[:3]
        nn_ = (LF @ feats(c[None])[0]).numpy(); near = np.argsort(-nn_)[:5]
        rows.append(dict(y=int(y), x=int(x), s=int(s), support=round(float(sup), 2), nonletter=round(float(p[-1]), 2),
                         top=[(I.CLASSES[t], round(float(p[t]), 2)) for t in top],
                         nearest=[(str(LL[j]), round(float(nn_[j]), 3)) for j in near], _near=near, _crop=(y, x, s)))
    rows.sort(key=lambda r: -(r["top"][0][1] * (1 - r["nonletter"]) * r["support"]))
    T = 128; Wd = 2 * T + 330 + 5 * (T + 4)
    g = Image.new("RGB", (Wd, 40 + len(rows[:20]) * (T + 8)), "white"); d = ImageDraw.Draw(g)
    d.text((6, 8), f"{scroll} {mesh}: moyenne de {len(fw)} cartes | lettres sures: carte {len(res['fwd'])} / temoin {len(res['ctl'])}", fill=(0, 0, 0), font=font)
    for i, r in enumerate(rows[:20]):
        y, x, s = r["_crop"]; y0, x0 = max(0, y - s // 4), max(0, x - s // 4); S2 = int(s * 1.5)
        Y = 40 + i * (T + 8)
        g.paste(Image.fromarray((avg[y0:y0 + S2, x0:x0 + S2] * 255).astype(np.uint8)).resize((T, T)).convert("RGB"), (0, Y))
        g.paste(Image.fromarray(ct[y0:y0 + S2, x0:x0 + S2]).resize((T, T)).convert("RGB"), (T + 4, Y))
        txt = "  ".join(f"{c} {q:.2f}" for c, q in r["top"])
        d.text((2 * T + 14, Y + 20), txt, fill=(180, 0, 0), font=font)
        d.text((2 * T + 14, Y + 55), f"non-lettre {r['nonletter']:.2f}  appui {r['support']}", fill=(60, 60, 60), font=small)
        d.text((2 * T + 14, Y + 80), "plus proches: " + " ".join(c for c, _ in r["nearest"]), fill=(0, 0, 160), font=small)
        for j, k in enumerate(r["_near"]):
            g.paste(Image.fromarray(LG[k]).resize((T, T)).convert("RGB"), (2 * T + 330 + j * (T + 4), Y))
    g.save(f"hunt/proposals/{scroll}_{mesh}.png")
    json.dump([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows], open(f"hunt/proposals/{scroll}_{mesh}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(mesh, "spots", len(rows), "sure letters fwd", len(res["fwd"]), "ctl", len(res["ctl"]), [r["top"][0] for r in rows[:12]], flush=True)
