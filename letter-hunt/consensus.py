"""Two-model consensus image: keep only what two independently trained ink models both draw.
Both maps are rank-normalised over the papyrus (so brightness/contrast differences are removed), then combined by the
geometric mean, so a stroke survives only if BOTH models rank it high. Also writes model B histogram-matched to model A
(same grey values, B's own ordering) to show how much of a visible difference is only contrast.
usage: consensus.py MAP_A.png MAP_B.png OUT_PREFIX"""
import sys
import numpy as np
from PIL import Image
a = np.asarray(Image.open(sys.argv[1]).convert("L"), np.float32); b = np.asarray(Image.open(sys.argv[2]).convert("L"), np.float32)
m = (a > 0) & (b > 0)
def rank(x):
    o = np.zeros_like(x); v = x[m]; o[m] = (np.argsort(np.argsort(v)) + 0.5) / len(v); return o
ra, rb = rank(a), rank(b)
srt = np.sort(a[m]); bm = np.zeros_like(b); bm[m] = srt[np.clip((rb[m] * len(srt)).astype(int), 0, len(srt) - 1)]
cons = np.clip((np.sqrt(ra * rb) - 0.6) / 0.35, 0, 1) * 255 * m
Image.fromarray(bm.astype(np.uint8)).save(sys.argv[3] + "_B_matched_to_A.png")
Image.fromarray(cons.astype(np.uint8)).save(sys.argv[3] + "_consensus.png")
print("rank correlation A/B:", round(float(np.corrcoef(ra[m], rb[m])[0, 1]), 3))
