"""Bench of REAL ink-map letters read by papyrologists (dev set for the Interprete, used to choose recipes):
 - PHerc0306B line 1 (First Letters 2025, Prodigle team, reading by N. Bodill): 15 letters, positions in mm from
   their evidence grid, cut from claim_map_fwd_fin.npy (CC BY-NC 4.0)
 - 2023 First Letters word PORPHYRAS (L. Farritor image with papyrologist boxes): 8 letters
usage: letter_bench.py [weights]  -> exact / top-3 per source"""
import sys
import numpy as np
from PIL import Image
import interprete as I

K = 115.741
L306 = [("Τ", -11.7, -9.5), ("Ο", -9.5, -7.4), ("Δ", -7.4, -4.0), ("Ι", -4.0, -2.9), ("Α", -2.9, -1.3), ("Υ", -1.3, 2.4),
        ("Τ", 2.4, 4.4), ("Ο", 4.4, 6.7), ("Α", 6.7, 9.3), ("Ι", 9.3, 11.1), ("Ρ", 11.1, 12.9), ("Ε", 12.9, 14.9),
        ("Τ", 14.9, 17.7), ("Ο", 17.7, 19.8), ("Ν", 19.8, 22.5)]
LPOR = [("Π", 150, 290), ("Ο", 290, 360), ("Ρ", 360, 440), ("Φ", 440, 550), ("Υ", 550, 615), ("Ρ", 615, 700), ("Α", 700, 800), ("Ϲ", 800, 880)]


def square64(c):
    h, w = c.shape; s = max(h, w); p = np.zeros((s, s), np.float32); p[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = c
    return np.asarray(Image.fromarray((np.clip(p, 0, 1) * 255).astype(np.uint8)).resize((64, 64)), np.float32) / 255


def bench():
    m = np.nan_to_num(np.load("data/letters_ext/pherc0306b/claim_map_fwd_fin.npy").astype(np.float32))
    hi = np.percentile(m[2300:2760, int(21 * K):int(59 * K)], 99.5)
    a = [(t, square64(m[2300:2760, int((u0 + 36) * K):int((u1 + 36) * K)][::-1] / hi)) for t, u0, u1 in L306]
    im = np.asarray(Image.open("data/letters_ext/firstletters2023/luke-boxes.png").convert("RGB")).astype(np.float32)
    g = 255 - im.mean(2); g[(np.abs(im[..., 0] - im[..., 1]) > 60) | (np.abs(im[..., 1] - im[..., 2]) > 60)] = 0
    b = [(t, square64(g[195:345, x0:x1] / 255)) for t, x0, x1 in LPOR]
    return {"PHerc0306B": a, "PORPHYRAS": b}


if __name__ == "__main__":
    it = I.Interprete(*(sys.argv[1:2] or []))
    tot = [0, 0, 0]
    for src, items in bench().items():
        P = it.predict(np.array([x for _, x in items]))
        ranks = [list(np.argsort(-p)).index(I.CLASSES.index(t)) for (t, _), p in zip(items, P)]
        ex, t3 = sum(r == 0 for r in ranks), sum(r < 3 for r in ranks)
        tot[0] += ex; tot[1] += t3; tot[2] += len(items)
        print(f"{src}: exact {ex}/{len(items)}  top3 {t3}/{len(items)}  read as: {''.join(I.CLASSES[int(np.argmax(p))][0] for p in P)}")
    print(f"TOTAL exact {tot[0]}/{tot[2]}  top3 {tot[1]}/{tot[2]}")
