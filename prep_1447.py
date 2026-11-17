"""Second never-seen native test: PHerc1447 w058 / w060 (8.64 um, 116 keV, 24-layer native renders) with refined ink
labels and certainty masks from YoussefMoNader/ink-8um-pherc1447-surfaces (CC BY-NC 4.0, evaluation only).
Writes data/pherc1447-<w>-native[-rev]/ in the segment format (the two layer orders; the physical one is the one where
every model scores far above 0.5). usage: prep_1447.py [w058 w060]"""
import io, json, os, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np, zarr
from numcodecs import Blosc
from PIL import Image
import prep
Image.MAX_IMAGE_PIXELS = None
HF = "https://huggingface.co/datasets/YoussefMoNader/ink-8um-pherc1447-surfaces/resolve/main/"


def img(path):
    with urllib.request.urlopen(HF + path, timeout=600) as r:
        return np.asarray(Image.open(io.BytesIO(r.read())))


for w in (sys.argv[1:] or ["w058", "w060"]):
    with ThreadPoolExecutor(8) as ex:
        layers = list(ex.map(lambda i: img(f"{w}/layers/{i:02d}.tif"), range(24)))
    vol = np.stack(layers).astype(np.uint8)
    lab = img(f"{w}/labels/inklabels.png"); msk = img(f"{w}/labels/mask.png")
    lab = lab[..., 0] if lab.ndim == 3 else lab; msk = msk[..., 0] if msk.ndim == 3 else msk
    ink, sup = (lab > 127).astype(np.uint8), ((msk > 0) & (vol[12] > 0)).astype(np.uint8)
    for tag, v in (("", vol), ("-rev", vol[::-1])):
        d = os.path.join(prep.ROOT, f"pherc1447-{w}-native{tag}"); os.makedirs(d, exist_ok=True)
        z = zarr.open(os.path.join(d, "volume.zarr"), mode="w", shape=v.shape, chunks=(v.shape[0], 128, 128), dtype="u1",
                      compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE))
        z[:] = np.ascontiguousarray(v)
        np.savez_compressed(os.path.join(d, "labels.npz"), ink=ink, sup=sup, val=sup.copy(), attrs=json.dumps({"source": HF + w}))
        json.dump(dict(segment=d, shape=list(v.shape), ink_px=int(ink.sum()), sup_px=int(sup.sum())), open(os.path.join(d, "info.json"), "w"))
    print(w, vol.shape, int(ink.sum()), int(sup.sum()), flush=True)
