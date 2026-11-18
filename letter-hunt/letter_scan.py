"""Letter scan: slide letter-sized windows over an ink map AND over its control map (reversed / shuffled layers),
ask the Interprete what each window is, keep windows read as a letter with confidence, and compare counts
forward vs control. A real text area gives many more confident letters on the forward map than on the control.
usage (library): scan(map, control, valid, px_mm, it) -> dict(fwd=[...], ctl=[...]) of (y, x, size, letter, p)
       letter_scan.py careful   -> runs over hunt/careful/*_maps.npz, writes hunt/letterscan/"""
import glob, json, os, sys
import numpy as np
from PIL import Image
import interprete as I

OUT = "hunt/letterscan"


def windows(m, valid, px_mm, sizes_mm=(2.4, 3.2, 4.2), stride=0.25):
    out = []
    for smm in sizes_mm:
        s = int(smm * px_mm); st = max(4, int(s * stride))
        for y in range(0, m.shape[0] - s + 1, st):
            for x in range(0, m.shape[1] - s + 1, st):
                if valid[y:y + s, x:x + s].mean() > 0.8:
                    out.append((y, x, s))
    return out


def crop64(m, y, x, s):
    c = m[y:y + s, x:x + s].astype(np.float32)
    return np.asarray(Image.fromarray((np.clip(c, 0, 1) * 255).astype(np.uint8)).resize((64, 64)), np.float32) / 255


def scan(m, ctl, valid, px_mm, it, p_min=0.5, ink_min=0.04):
    W = windows(m, valid, px_mm); res = {}
    for name, mm in (("fwd", m), ("ctl", ctl)):
        hits = []
        for k in range(0, len(W), 4096):
            part = W[k:k + 4096]
            X = np.array([crop64(mm, *w) for w in part])
            ink = X.reshape(len(X), -1).mean(1)
            P = it.predict(X)
            for (y, x, s), p, a in zip(part, P, ink):
                c = int(np.argmax(p[:-1]))
                if a >= ink_min and p[c] >= p_min and p[-1] < 0.2:
                    hits.append((y, x, s, I.CLASSES[c], float(p[c])))
        res[name] = nms(hits)
    return res


def nms(hits, iou=0.3):
    hits = sorted(hits, key=lambda h: -h[4]); keep = []
    for h in hits:
        y, x, s = h[:3]
        if all(_iou((y, x, s), k[:3]) < iou for k in keep):
            keep.append(h)
    return keep


def _iou(a, b):
    y0, x0 = max(a[0], b[0]), max(a[1], b[1]); y1, x1 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[2], b[1] + b[2])
    inter = max(0, y1 - y0) * max(0, x1 - x0); return inter / (a[2] ** 2 + b[2] ** 2 - inter)


def montage(m, hits, path, n=40):
    tiles = []
    for y, x, s, c, p in hits[:n]:
        t = Image.fromarray((np.clip(m[y:y + s, x:x + s], 0, 1) * 255).astype(np.uint8)).resize((96, 96)).convert("RGB")
        from PIL import ImageDraw
        ImageDraw.Draw(t).text((3, 2), f"{c} {p:.2f}", fill=(255, 60, 60)); tiles.append(t)
    if not tiles:
        return
    g = Image.new("RGB", (10 * 98, -(-len(tiles) // 10) * 98))
    for k, t in enumerate(tiles):
        g.paste(t, ((k % 10) * 98, (k // 10) * 98))
    g.save(path)


if __name__ == "__main__" and sys.argv[1:2] == ["careful"]:
    os.makedirs(OUT, exist_ok=True)
    it = I.Interprete(*(sys.argv[2:3] or []))
    summary = []
    for f in sorted(glob.glob("hunt/careful/*_maps.npz")):
        z = np.load(f); m, r = z["cons"] / 255.0, z["rev"] / 255.0; valid = z["ct"] > 0
        cm2 = valid.sum() / (107.0 * 10) ** 2
        res = scan(m, r, valid, 107.0, it)
        tag = os.path.basename(f)[:-9]
        montage(m, res["fwd"], f"{OUT}/{tag}_fwd.png"); montage(r, res["ctl"], f"{OUT}/{tag}_ctl.png")
        rec = dict(tag=tag, cm2=round(cm2, 2), fwd=len(res["fwd"]), ctl=len(res["ctl"]),
                   fwd_per_cm2=round(len(res["fwd"]) / max(cm2, 1e-3), 1), ctl_per_cm2=round(len(res["ctl"]) / max(cm2, 1e-3), 1),
                   top=[h[3] + f"{h[4]:.2f}" for h in res["fwd"][:12]], hits=res["fwd"])
        summary.append(rec); print({k: rec[k] for k in ("tag", "cm2", "fwd", "ctl", "fwd_per_cm2", "ctl_per_cm2", "top")}, flush=True)
    json.dump(summary, open(f"{OUT}/careful_summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
