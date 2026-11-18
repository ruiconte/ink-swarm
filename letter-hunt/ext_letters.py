"""External letter library from ICDAR2023 HomerComp (Greek papyri from Egypt, same script family as Herculaneum):
every annotated letter box is cropped, ink binarised (dark ink on papyrus, Otsu inside the box), normalised to a
64x64 mask like hunt/letter_library.npz, and saved with its Greek letter label and the grey crop.
usage: ext_letters.py  -> hunt/letters_homercomp.npz + .png overview"""
import json, os, glob
import numpy as np
from PIL import Image
from skimage.filters import threshold_otsu
from scipy import ndimage


def norm(mask, S=64):                      # same normalisation as letter_library.norm (not imported: that module runs on import)
    yy, xx = np.nonzero(mask); c = mask[yy.min():yy.max() + 1, xx.min():xx.max() + 1]; h, w = c.shape; s = max(h, w)
    pad = np.zeros((s, s), bool); pad[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = c
    return np.asarray(Image.fromarray(pad.astype(np.uint8) * 255).resize((S, S))) > 127


Image.MAX_IMAGE_PIXELS = None
from PIL import ImageFile; ImageFile.LOAD_TRUNCATED_IMAGES = True
ROOT = "data/letters_ext/homercomp/hc"
SETS = [("2a.Training/HomerCompTraining.json", "2a.Training/images", None), ("2c.Answers/AnswersToTesting.json", "2b.Testing/images", "2b.Testing/HomerCompTesting.json")]
masks, greys, labels, src = [], [], [], []
for js, imdir, imjs in SETS:
    d = json.load(open(os.path.join(ROOT, js), encoding="utf-8"))
    cat = {c["id"]: c.get("name", str(c["id"])) for c in d["categories"]}
    files = {}
    for p in glob.glob(os.path.join(ROOT, imdir, "**", "*.*"), recursive=True):
        files[os.path.basename(p)] = p
    ims = {i["id"]: i for i in (json.load(open(os.path.join(ROOT, imjs), encoding="utf-8")) if imjs else d)["images"]}
    ims.update({i["id"]: i for i in d.get("images", [])})
    by_im = {}
    for a in d["annotations"]:
        by_im.setdefault(a["image_id"], []).append(a)
    for iid, anns in by_im.items():
        fn = os.path.basename(ims.get(iid, {}).get("file_name", ""))
        if fn not in files:
            continue
        g = np.asarray(Image.open(files[fn]).convert("L"))
        for a in anns:
            x, y, w, h = map(int, a["bbox"])
            if w < 8 or h < 8:
                continue
            c = g[max(y, 0):y + h, max(x, 0):x + w]
            if c.size < 64 or c.std() < 5:
                continue
            m = c < threshold_otsu(c)
            m = ndimage.binary_opening(m)
            if m.sum() < 0.05 * m.size or m.sum() > 0.8 * m.size:
                continue
            masks.append(norm(m)); labels.append(cat.get(a["category_id"], "?")); src.append(fn)
            greys.append(np.asarray(Image.fromarray(c).resize((64, 64))))
masks, greys = np.array(masks), np.array(greys, np.uint8)
np.savez_compressed("hunt/letters_homercomp.npz", shapes=masks, greys=greys, labels=np.array(labels), src=np.array(src))
print(len(masks), "letters;", len(set(labels)), "labels;", len(set(src)), "papyri")
from collections import Counter
print(Counter(labels).most_common(30))
grid = Image.new("L", (32 * 66, 20 * 66), 0)
idx = np.random.default_rng(0).choice(len(masks), min(640, len(masks)), replace=False)
for k, i in enumerate(idx):
    grid.paste(Image.fromarray(greys[i]), ((k % 32) * 66, (k // 32) * 66))
grid.save("hunt/letters_homercomp.png")
