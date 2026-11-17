"""Native 8-9 um training data for Argus V2: domenicor046/reader-v2-labels (MIT; dense labels of 70 segments on the
native 113/116/53 keV scans) + the public native surface volumes, written in train.py's segment format as
data/n<object>-<segment>/{volume.zarr, labels.npz, done_tiles.npy, info.json}.
To keep each rented node's download small, only a random share of the supervised area is fetched, in blocks of
BLOCKxBLOCK 128 px tiles (plus a one-tile margin so 256 px crops never run into missing CT).
Segments named in --val get val = sup (used only for checkpoint selection; PHerc0841 is never in these labels).
usage: native_v2_prep.py [--frac 0.2] [--block 6] [--objects 0139,0814,...] [--val seg,seg]"""
import argparse, csv, io, json, os, tarfile, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import zarr
from numcodecs import Blosc
from scipy.ndimage import binary_dilation

ROOT = os.environ.get("INK_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
HF = "https://huggingface.co/datasets/domenicor046/reader-v2-labels/resolve/main/"
S3 = "https://vesuvius-challenge-open-data.s3.amazonaws.com/"
VAL = "20260302000001-w050_2026030220,20250511200236,20250925212701-auto_grown_20250925212701"


def get(url, tries=6):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(2 * (i + 1))
        except Exception:
            time.sleep(2 * (i + 1))
    raise RuntimeError(url)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, default=0.2)
    ap.add_argument("--block", type=int, default=6)
    ap.add_argument("--objects", default="0139,0814,0500P2,0009B,0343P,0172")
    ap.add_argument("--val", default=VAL)
    ap.add_argument("--labels_dir", default="/tmp/rv2")
    ap.add_argument("--seed", type=int, default=-1, help="-1: derived from RUN_NAME, so each rented node fetches a different share")
    a = ap.parse_args()
    t0 = time.time()
    lab_root = os.path.join(a.labels_dir, "reader-v2-labels")
    if not os.path.isdir(lab_root):
        os.makedirs(a.labels_dir, exist_ok=True)
        with urllib.request.urlopen(HF + "reader-v2-dense-native-labels.tar", timeout=600) as r:
            tarfile.open(fileobj=r, mode="r|").extractall(a.labels_dir)
        print(f"labels extracted in {time.time() - t0:.0f} s", flush=True)
    rows = list(csv.DictReader(io.StringIO(get(HF + "manifest.csv").decode())))
    vals = set(a.val.split(","))
    objs = set(a.objects.split(","))
    import zlib
    seed = a.seed if a.seed >= 0 else zlib.crc32(os.environ.get("RUN_NAME", "x").encode())
    print("block sampling seed", seed, flush=True)
    rng = np.random.default_rng(seed)
    tot = 0
    for row in rows:
        if row["object"] not in objs:
            continue
        seg, obj = row["segment"], row["object"]
        name = f"n{obj.lower()}-{seg}"
        d = os.path.join(ROOT, name)
        if os.path.exists(os.path.join(d, "done_tiles.npy")):
            continue
        ld = os.path.join(lab_root, obj, seg)
        z = int(row["label_z"])
        ink = np.asarray(zarr.open(os.path.join(ld, f"{seg}_inklabels.zarr"), mode="r")["0"][z]) > 0
        sup = np.asarray(zarr.open(os.path.join(ld, f"{seg}_supervision_mask.zarr"), mode="r")["0"][z]) > 0
        url = S3 + row["surface_volume_s3"].rstrip("/")
        za = json.loads(get(f"{url}/0/.zarray"))
        Z, H, W = za["shape"]; cz, cy, cx = za["chunks"]; sep = za.get("dimension_separator", ".")
        assert (H, W) == sup.shape and cz == Z, (name, za["shape"], sup.shape)
        T, B = cy, a.block
        ny, nx = -(-H // T), -(-W // T)
        has = np.zeros((ny, nx), bool)
        ys, xs = np.nonzero(sup[::8, ::8]); has[ys * 8 // T, xs * 8 // T] = True
        is_val = seg in vals
        frac = min(1.0, a.frac * 2.5) if is_val else a.frac
        blocks = [(by, bx) for by in range(0, ny, B) for bx in range(0, nx, B) if has[by:by + B, bx:bx + B].any()]
        keep = [b for b in blocks if rng.random() < frac] or blocks[:1]
        core = np.zeros((ny, nx), bool)
        for by, bx in keep:
            core[by:by + B, bx:bx + B] = True
        core &= has
        fetch = binary_dilation(core, np.ones((3, 3), bool))
        m = np.kron(core, np.ones((T, T), bool))[:H, :W]
        sup &= m; ink &= sup
        os.makedirs(d, exist_ok=True)
        vol = zarr.open(os.path.join(d, "volume.zarr"), mode="w", shape=(Z, H, W), chunks=(Z, T, T), dtype="u1",
                        compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE), fill_value=0)
        comp = za.get("compressor")
        import numcodecs
        codec = numcodecs.get_codec(dict(comp)) if comp else None

        def one(k):
            iy, ix = k
            raw = get(f"{url}/0/0{sep}{iy}{sep}{ix}")
            if raw is None:
                return 0
            raw = codec.decode(raw) if codec else raw
            arr = np.frombuffer(raw, np.uint8).reshape(cz, cy, cx)
            y0, x0 = iy * T, ix * T
            vol[:, y0:min(y0 + T, H), x0:min(x0 + T, W)] = arr[:, :min(T, H - y0), :min(T, W - x0)]
            return len(raw)
        with ThreadPoolExecutor(32) as ex:
            nb = sum(ex.map(one, list(zip(*np.nonzero(fetch)))))
        tot += nb
        val = sup.copy() if is_val else np.zeros(0, np.uint8)
        np.savez_compressed(os.path.join(d, "labels.npz"), ink=ink.astype(np.uint8), sup=sup.astype(np.uint8),
                            val=val.astype(np.uint8) if is_val else val, attrs=json.dumps({"source": url, "object": obj}))
        info = dict(segment=name, object=obj, source=url, shape=[Z, H, W], tiles=int(fetch.sum()), val=is_val,
                    ink_px=int(ink.sum()), sup_px=int(sup.sum()), download_gb=round(nb / 1e9, 2))
        json.dump(info, open(os.path.join(d, "info.json"), "w"), indent=1)
        np.save(os.path.join(d, "done_tiles.npy"), fetch)
        print(name, info["tiles"], "tiles", info["download_gb"], "GB", f"{time.time() - t0:.0f} s", flush=True)
    print(f"native data ready: {tot / 1e9:.1f} GB in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
