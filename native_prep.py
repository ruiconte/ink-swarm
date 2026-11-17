"""Native 9.4 um / 113 keV version of the never-seen PHerc0841 w00 test (reader-v3's home scan format):
the native surface volume rendered on the same mesh, and the team's ink labels resampled from the 2.4 um level-2 grid
(4115 x 4640) onto the native grid (4220 x 4760, x1.0255) by nearest neighbour. usage: native_prep.py"""
import json, os
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import zarr
from numcodecs import Blosc
import prep

URL = ("https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0841/segments/20260220213127-w00/surface-volumes/"
       "9.366um-1.2m-113keV-volume-20250821151531.zarr")
SRC = "pherc0841-w00"
NAME = "pherc0841-w00-native"


def main():
    d = os.path.join(prep.ROOT, NAME); os.makedirs(d, exist_ok=True)
    za = prep.get(f"{URL}/0/.zarray").json()
    Z, H, W = za["shape"]; cz, cy, cx = za["chunks"]; sep = za.get("dimension_separator", ".")
    L = np.load(os.path.join(prep.ROOT, SRC, "labels.npz"))
    h0, w0 = L["ink"].shape
    yi = np.minimum((np.arange(H) * h0 / H).astype(int), h0 - 1)
    xi = np.minimum((np.arange(W) * w0 / W).astype(int), w0 - 1)
    ink, sup = L["ink"][yi][:, xi], L["sup"][yi][:, xi]
    T = cy
    tiles = np.zeros((-(-H // T), -(-W // T)), bool)
    ys, xs = np.nonzero(sup); tiles[ys // T, xs // T] = True
    from scipy.ndimage import binary_dilation
    tiles = binary_dilation(tiles, np.ones((3, 3), bool))
    vol = zarr.open(os.path.join(d, "volume.zarr"), mode="w", shape=(Z, H, W), chunks=(Z, T, T), dtype="u1",
                    compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE), fill_value=0)

    def one(k):
        iy, ix = k
        r = prep.get(f"{URL}/0/0{sep}{iy}{sep}{ix}")
        if r is None:
            return
        a = np.frombuffer(r.content, np.uint8).reshape(cz, cy, cx)
        y0, x0 = iy * T, ix * T
        vol[:, y0:min(y0 + T, H), x0:min(x0 + T, W)] = a[:, :min(T, H - y0), :min(T, W - x0)]
    with ThreadPoolExecutor(24) as ex:
        list(ex.map(one, list(zip(*np.nonzero(tiles)))))
    np.savez_compressed(os.path.join(d, "labels.npz"), ink=ink, sup=sup, val=sup.copy(), attrs=json.dumps({"source": URL}))
    json.dump(dict(segment=NAME, source=URL, shape=[Z, H, W], ink_px=int(ink.sum()), sup_px=int(sup.sum())),
              open(os.path.join(d, "info.json"), "w"), indent=1)
    print(NAME, Z, H, W, int(ink.sum()), int(sup.sum()))


if __name__ == "__main__":
    main()
