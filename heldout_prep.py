"""Held-out test data in exactly our training input format (2.4 um surface volume, level 2, z 13..97 mean-pooled by 4
-> 21 slices), for the checks the community runs before trusting an ink model:
  pherc0841-w00   : a scroll NO model of ours saw, with the team's ink labels (2026-09-18)
  pherc0139-w053  : blank papyrus (no text): every pixel marked as ink is a false alarm
Writes data/<name>/{volume.zarr, labels.npz, info.json} like prep.py so train.Segment can load them.
usage: heldout_prep.py [name ...]"""
import json, os, struct, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import numcodecs
import zarr
from numcodecs import Blosc

import prep

B = "https://vesuvius-challenge-open-data.s3.amazonaws.com/"
SETS = {
    "pherc0841-w00": dict(
        vol=B + "PHerc0841/segments/20260220213127-w00/surface-volumes/2.403um-0.22m-77keV-volume-20260319124803.zarr",
        lab=B + "PHerc0841/segments/20260220213127-w00/ink-labels/2.403um-volume-20260319124803/20260918"),
    "pherc0139-w053": dict(
        vol=B + "PHerc0139/segments/20260220000000-w053_2026022021/surface-volumes/2.399um-0.22m-78keV-volume-20260102150214.zarr",
        lab=None, crop=3072),
}
# extra blank inner wraps of PHerc0139 (text ends before w053): candidates for blank hard negatives in TRAINING.
# w052/w054 (direct neighbours of the w053 test) are deliberately not used.
for _w in ("20260115000001-w056_2026011514", "20260127000000-w057_2026012713", "20260210000000-w058_2026021020"):
    SETS["pherc0139-" + _w.split("-")[1][:4]] = dict(
        vol=B + f"PHerc0139/segments/{_w}/surface-volumes/2.399um-0.22m-78keV-volume-20260102150214.zarr", lab=None, crop=2048)
LEVEL, Z0, Z1, POOL = 2, 13, 97, 4


def read_v3_shard_2d(url):
    """level-2 array of a zarr v3 2D label (one sharded chunk, inner chunks 512x512, blosc, index at end)"""
    meta = prep.get(f"{url}/{LEVEL}/zarr.json").json()
    H, W = meta["shape"]
    sh = meta["codecs"][0]["configuration"]
    iy, ix = sh["chunk_shape"]
    SY, SX = meta["chunk_grid"]["configuration"]["chunk_shape"]
    raw = prep.get(f"{url}/{LEVEL}/c/0/0")
    out = np.zeros((H, W), np.uint8)
    if raw is None:
        return out
    buf = raw.content
    ny, nx = -(-SY // iy), -(-SX // ix)
    idx = buf[-(ny * nx * 16 + 4):-4]
    codec = Blosc(cname="zstd")
    for k in range(ny * nx):
        off, nb = struct.unpack_from("<QQ", idx, k * 16)
        if off == 2 ** 64 - 1:
            continue
        a = np.frombuffer(codec.decode(buf[off:off + nb]), np.uint8).reshape(iy, ix)
        y0, x0 = (k // nx) * iy, (k % nx) * ix
        if y0 >= H or x0 >= W:
            continue
        out[y0:min(y0 + iy, H), x0:min(x0 + ix, W)] = a[:min(iy, H - y0), :min(ix, W - x0)]
    return out


def build(name):
    cfg = SETS[name]
    d = os.path.join(prep.ROOT, name)
    os.makedirs(d, exist_ok=True)
    za = prep.get(f"{cfg['vol']}/{LEVEL}/.zarray").json()
    Z, H, W = za["shape"]
    cz, cy, cx = za["chunks"]
    assert Z == 109 and cz == Z and cy == prep.TILE, za
    T = prep.TILE
    ny, nx = -(-H // T), -(-W // T)
    if cfg["lab"]:
        ink = read_v3_shard_2d(cfg["lab"] + "/inklabels.zarr")
        sup = read_v3_shard_2d(cfg["lab"] + "/supervision.zarr")
        assert ink.shape == (H, W), (ink.shape, H, W)
        val = sup.copy()                      # whole scroll is held out: score on every supervised pixel
        tiles = np.zeros((ny, nx), bool)
        ys, xs = np.nonzero(sup)
        tiles[ys // T, xs // T] = True
    else:                                     # blank papyrus: a central crop, labels all "no ink"
        c = cfg["crop"] // T
        tiles = np.zeros((ny, nx), bool)
        y0, x0 = max(ny // 2 - c // 2, 0), max(nx // 2 - c // 2, 0)
        tiles[y0:y0 + c, x0:x0 + c] = True
        ink = np.zeros((H, W), np.uint8)
        sup = val = None
    from scipy.ndimage import binary_dilation
    tiles = binary_dilation(tiles, np.ones((3, 3), bool))
    codec = numcodecs.get_codec(dict(za["compressor"])) if za["compressor"] else None
    sep = za.get("dimension_separator", ".")
    depth = (Z1 - Z0) // POOL
    vol = zarr.open(os.path.join(d, "volume.zarr"), mode="a", shape=(depth, H, W), chunks=(depth, T, T), dtype="u1",
                    compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE), fill_value=0)
    todo = list(zip(*np.nonzero(tiles)))

    def one(k):
        iy, ix = k
        r = prep.get(f"{cfg['vol']}/{LEVEL}/0{sep}{iy}{sep}{ix}")
        if r is None:
            return
        raw = codec.decode(r.content) if codec else r.content
        a = np.frombuffer(raw, np.uint8).reshape(cz, cy, cx)[Z0:Z1].astype(np.float32)
        a = np.rint(a.reshape(depth, POOL, cy, cx).mean(1)).astype(np.uint8)
        y0, x0 = iy * T, ix * T
        vol[:, y0:min(y0 + T, H), x0:min(x0 + T, W)] = a[:, :min(T, H - y0), :min(T, W - x0)]
    t0 = time.time()
    with ThreadPoolExecutor(24) as ex:
        for i, _ in enumerate(ex.map(one, todo)):
            if i % 300 == 299:
                print(f"  {name}: {i + 1}/{len(todo)} tiles, {(i + 1) * 1.79 / (time.time() - t0):.0f} MB/s", flush=True)
    if sup is None:                           # blank: papyrus pixels of the crop (non-empty CT) are the region
        mid = np.asarray(vol[depth // 2])
        from scipy.ndimage import binary_erosion
        core = binary_erosion(tiles, np.ones((3, 3), bool))
        crop = np.kron(core, np.ones((T, T), bool))[:H, :W]
        sup = (crop & (mid > 0)).astype(np.uint8)
        val = sup.copy()
    if TRAIN and cfg["lab"] is None:          # training blank negative: no val region, all supervised pixels = no ink
        val = np.zeros(0, np.uint8)
        np.save(os.path.join(d, "done_tiles.npy"), tiles)
    np.savez_compressed(os.path.join(d, "labels.npz"), ink=ink, sup=sup, val=val, attrs=json.dumps(cfg))
    info = dict(segment=name, source=cfg["vol"], level=LEVEL, shape=[Z, H, W], z=[Z0, Z1], pool=POOL, depth=depth,
                tiles=len(todo), ink_px=int((ink > 0).sum()), sup_px=int((sup > 0).sum()))
    json.dump(info, open(os.path.join(d, "info.json"), "w"), indent=1)
    print(name, info, flush=True)


TRAIN = "--train" in sys.argv
if __name__ == "__main__":
    for n in ([x for x in sys.argv[1:] if x != "--train"] or list(SETS)):
        build(n)
