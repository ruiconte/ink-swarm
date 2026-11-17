"""Dense pseudo-labels (domenicor046/ink9um-dense-labels, MIT): for each of the 7 segments, store data/<seg>/dense.npz
(ink, supervision on the label grid) and download the volume tiles the dense supervision needs from the public S3
bucket (same 21-slice pooling as prep.py). Checks that the dense validation masks match ours exactly."""
import json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import zarr
import numcodecs
from numcodecs import Blosc

import prep

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "dense", "pseudo_labels", "aligned-scrollprizeorg-21slices")
DATA = prep.ROOT
TILE = 128


def plane(path):
    a = zarr.open(path, mode="r")
    arr = a["0"] if hasattr(a, "array_keys") and "0" in list(a.array_keys()) else a
    sub = np.asarray(arr[:, ::16, ::16])
    z = int(np.argmax([(sub[i] > 0).sum() for i in range(sub.shape[0])]))
    return np.asarray(arr[z])


def main():
    for seg in sorted(os.listdir(SRC)):
        d = os.path.join(DATA, seg)
        info = json.load(open(os.path.join(d, "info.json")))
        L = np.load(os.path.join(d, "labels.npz"))
        t0 = time.time()
        ink = plane(os.path.join(SRC, seg, f"{seg}_inklabels.zarr")) > 0
        sup = plane(os.path.join(SRC, seg, f"{seg}_supervision_mask.zarr")) > 0
        vp = os.path.join(SRC, seg, f"{seg}_validation_mask.zarr")
        ours_val = (L["val"] > 0) if L["val"].size else np.zeros_like(sup)
        if os.path.exists(vp):
            dval = plane(vp) > 0
            same = int((dval != ours_val).sum())
            print(f"{seg}: validation mask differs from ours on {same} px", flush=True)
        overlap = int((sup & ours_val).sum())
        if ours_val.any():
            # validation masks were revised after the dense labels were made: keep every validation pixel, plus a
            # 16 px margin, out of the training supervision so the exam stays clean
            from scipy.ndimage import binary_dilation
            guard = binary_dilation(ours_val, np.ones((33, 33), bool))
            sup &= ~guard
        print(f"{seg}: removed {overlap} overlapping px (+ margin) from dense supervision", flush=True)
        np.savez_compressed(os.path.join(d, "dense.npz"), ink=ink, sup=sup)
        H, W = sup.shape
        ty, tx = -(-H // TILE), -(-W // TILE)
        need = np.zeros((ty, tx), bool)
        ys, xs = np.nonzero(sup[::8, ::8])
        need[(ys * 8) // TILE, (xs * 8) // TILE] = True
        done = np.load(os.path.join(d, "done_tiles.npy"))
        todo = np.argwhere(need & ~done)
        url, level, za = prep.source_meta({"source_surface_volume": info["source"], "source_surface_volume_level": info["level"]})
        Z, H0, W0 = za["shape"]
        cz, cy, cx = za["chunks"]
        z0, z1 = info["z"]
        pool, depth = info["pool"], info["depth"]
        sep = za.get("dimension_separator", ".")
        codec = numcodecs.get_codec(dict(za["compressor"])) if za["compressor"] else None
        vol = zarr.open(os.path.join(d, "volume.zarr"), mode="a")

        def one(k):
            iy, ix = int(k[0]), int(k[1])
            r = prep.get(f"{url}/{level}/0{sep}{iy}{sep}{ix}")
            if r is None:
                return None
            raw = codec.decode(r.content) if codec else r.content
            x = np.frombuffer(raw, np.uint8).reshape(cz, cy, cx)[z0:z1].astype(np.float32)
            if pool > 1:
                x = x.reshape(depth, pool, cy, cx).mean(1)
            x = np.rint(x).astype(np.uint8)
            y0, x0 = iy * TILE, ix * TILE
            vol[:, y0:min(y0 + TILE, H0), x0:min(x0 + TILE, W0)] = x[:, :min(TILE, H0 - y0), :min(TILE, W0 - x0)]
            return k
        n = 0
        with ThreadPoolExecutor(24) as ex:
            for k in ex.map(one, todo):
                if k is not None:
                    done[int(k[0]), int(k[1])] = True
                    n += 1
                    if n % 500 == 0:
                        np.save(os.path.join(d, "done_tiles.npy"), done)
                        print(f"  {seg}: {n}/{len(todo)} tiles", flush=True)
        np.save(os.path.join(d, "done_tiles.npy"), done)
        print(f"{seg}: dense ink {int(ink.sum())} px, supervision {int(sup.sum())} px ({sup.mean():.0%} of canvas), "
              f"{n} new tiles, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
