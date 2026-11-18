"""AL-PUB v2 (Ancient Lives, Oxyrhynchus Greek papyri, crowd-labelled letter crops) -> letter library, read straight
from the Kaggle zip: 64x64 grey crop, Otsu ink mask normalised like hunt/letter_library.npz, letter label.
usage: alpub_letters.py PATH_TO_ZIP  -> hunt/letters_alpub.npz"""
import io, sys, zipfile
from collections import Counter
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.filters import threshold_otsu


def norm(mask, S=64):
    yy, xx = np.nonzero(mask); c = mask[yy.min():yy.max() + 1, xx.min():xx.max() + 1]; h, w = c.shape; s = max(h, w)
    pad = np.zeros((s, s), bool); pad[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = c
    return np.asarray(Image.fromarray(pad.astype(np.uint8) * 255).resize((S, S))) > 127


z = zipfile.ZipFile(sys.argv[1])
names = [n for n in z.namelist() if n.lower().endswith((".jpg", ".png")) and "/images/" in n]
masks, greys, labels, src = [], [], [], []
for k, n in enumerate(names):
    try:
        c = np.asarray(Image.open(io.BytesIO(z.read(n))).convert("L"))
    except Exception:
        continue
    if c.size < 64 or c.std() < 5:
        continue
    m = ndimage.binary_opening(c < threshold_otsu(c))
    if m.sum() < 0.05 * m.size or m.sum() > 0.8 * m.size:
        continue
    masks.append(norm(m)); greys.append(np.asarray(Image.fromarray(c).resize((64, 64))))
    labels.append(n.split("/")[-2]); src.append(n.split("/")[-1].split("_")[1])
    if k % 20000 == 0:
        print(k, "/", len(names), flush=True)
np.savez_compressed("hunt/letters_alpub.npz", shapes=np.array(masks), greys=np.array(greys, np.uint8),
                    labels=np.array(labels), src=np.array(src))
print(len(masks), "letters kept of", len(names), ";", len(set(src)), "papyri")
print(Counter(labels).most_common())
