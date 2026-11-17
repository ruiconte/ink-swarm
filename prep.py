"""Training data for our own ink-detection models (Vesuvius Challenge, 9 um corpus "ink_9um").
Labels come from the public Hugging Face bucket scrollprize/datasets (ink_9um); each label's .zattrs names its
source surface volume on the public S3 open-data bucket, the pyramid level and the Z planes to pool.
Only the 128x128 XY tiles that carry supervision (plus a one-tile margin for context) are downloaded; each source
chunk is reduced to the 21-slice representation right away (planes z0..z1, mean of 4, half-to-even rounding,
same as the reference preparer) and stored compressed in a local zarr.
usage: prep.py list | prep.py plan | prep.py fetch [segment ...]"""
import json, os, random, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import requests
import zarr
import numcodecs
from numcodecs import Blosc

HF_TREE = "https://huggingface.co/api/buckets/scrollprize/datasets/tree/ink_9um/labels"
HF_FILE = "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels"
ROOT = os.environ.get("INK_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
TILE = 128
S = requests.Session()
S.headers["User-Agent"] = "scroll-ink-prep"
# optional Hugging Face token (read-only) from the environment or ../scroll-ink/.env: higher download rate limit
_tok = os.environ.get("HF_TOKEN")
if not _tok and os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")):
    for _l in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"), encoding="utf-8"):
        if _l.startswith("HF_TOKEN="):
            _tok = _l.split("=", 1)[1].strip()
HF_AUTH = {"Authorization": f"Bearer {_tok}"} if _tok else {}   # sent to Hugging Face only: S3 refuses foreign auth


def get(url, tries=12, **kw):
    for i in range(tries):
        try:
            r = S.get(url, timeout=120, headers=HF_AUTH if "huggingface.co" in url else None, **kw)
            if r.status_code == 404:
                return None
            if r.status_code == 429:          # Hugging Face rate limit: wait as asked, then retry
                time.sleep(float(r.headers.get("Retry-After", 30)) + random.random() * 5)
                continue
            r.raise_for_status()
            return r
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"gave up on {url}")


def tree(prefix):
    """every file path under labels/<prefix> (paginated)"""
    out, url, params = [], f"{HF_TREE}/{prefix}", {"recursive": "true", "limit": 1000}
    while url:
        r = get(url, params=params)
        out += [it["path"] for it in r.json() if it["type"] == "file"]
        nxt = r.links.get("next", {}).get("url")
        url, params = nxt, None
    return out


README = "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/README.md"
NATIVE_SOURCE = {}


def segments():
    """(family, segment) from the two tables of the dataset README (the bucket API only lists files, by the 100k)"""
    import re
    text = get(README).text
    out = []
    head, _, native = text.partition("Native 9.362")
    for part, fam in ((head, "aligned-scrollprizeorg-21slices"), (native, "native9-scrollprizeorg-21slices")):
        for seg, path in re.findall(r"^\|\s*([\w-]+)\s*\|[^|]*\|\s*\[[^\]]*\]\(https://vesuvius-challenge-open-data"
                                    r"\.s3\.amazonaws\.com/index\.html#([^)]+)\)", part, re.M):
            out.append((fam, seg))
            if fam.startswith("native"):
                NATIVE_SOURCE[seg] = "https://vesuvius-challenge-open-data.s3.amazonaws.com/" + path.rstrip("/")
    return out


def read_label(fam, seg, kind, level=0, tiles=None):
    """2D annotated plane of <seg>_<kind>.zarr at `level`, or None. With `tiles` (bool grid of 128 px tiles at
    level 0), only those chunks are fetched: Hugging Face rate-limits anonymous downloads."""
    base = f"{HF_FILE}/{fam}/{seg}/{seg}_{kind}.zarr"
    za = get(f"{base}/{level}/.zarray")
    if za is None:
        return None, None
    za = za.json()
    attrs = get(f"{base}/.zattrs").json()
    zc = int(attrs.get("annotation_center_channel", za["shape"][0] // 2))
    sep = za.get("dimension_separator", ".")
    codec = numcodecs.get_codec(dict(za["compressor"])) if za["compressor"] else None
    D, H, W = za["shape"]
    cz, cy, cx = za["chunks"]
    out = np.zeros((H, W), np.uint8)
    zi = zc // cz
    # zarr stores only non-empty chunks: list them (1 request per 1000 files) instead of probing every chunk
    present = tree(f"{fam}/{seg}/{seg}_{kind}.zarr/{level}")
    keys = []
    for p in present:
        parts = p.rsplit(f"_{kind}.zarr/{level}/", 1)[-1].replace("/", ".").split(".")
        if len(parts) == 3 and parts[0] == str(zi):
            keys.append((int(parts[1]), int(parts[2])))

    def one(k):
        iy, ix = k
        r = get(f"{base}/{level}/{zi}{sep}{iy}{sep}{ix}")
        if r is None:
            return
        raw = codec.decode(r.content) if codec else r.content
        a = np.frombuffer(raw, np.uint8).reshape(cz, cy, cx)[zc % cz]
        y0, x0 = iy * cy, ix * cx
        out[y0:y0 + cy, x0:x0 + cx] = a[:min(cy, H - y0), :min(cx, W - x0)]
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(one, keys))
    return out, attrs


def needed_tiles(sup, val):
    m = (sup > 0) | (val > 0 if val is not None else False)
    H, W = m.shape
    ty, tx = -(-H // TILE), -(-W // TILE)
    t = np.zeros((ty, tx), bool)
    for iy in range(ty):
        for ix in range(tx):
            t[iy, ix] = m[iy * TILE:(iy + 1) * TILE, ix * TILE:(ix + 1) * TILE].any()
    from scipy.ndimage import binary_dilation
    return binary_dilation(t, np.ones((3, 3), bool))


def source_meta(attrs):
    url = (attrs.get("source_surface_volume") or NATIVE_SOURCE[attrs["case"]]).rstrip("/")
    level = int(attrs.get("source_surface_volume_level", 0))
    za = get(f"{url}/{level}/.zarray").json()
    return url, level, za


def fetch(fam, seg, plan_only=False):
    d = os.path.join(ROOT, seg)
    os.makedirs(d, exist_ok=True)
    lab_path = os.path.join(d, "labels.npz")
    if os.path.exists(lab_path):
        L = np.load(lab_path, allow_pickle=False)
        ink, sup = L["ink"], L["sup"]
        val = L["val"] if L["val"].size else None
        attrs = json.loads(str(L["attrs"]))
    else:
        ink, attrs = read_label(fam, seg, "inklabels")
        attrs.setdefault("case", seg)
        sup, _ = read_label(fam, seg, "supervision_mask")
        val, _ = read_label(fam, seg, "validation_mask")
        np.savez_compressed(lab_path, ink=ink, sup=sup, val=val if val is not None else np.zeros(0, np.uint8),
                            attrs=json.dumps(attrs))
    tiles = needed_tiles(sup, val)
    url, level, za = source_meta(attrs)
    Z, H, W = za["shape"]
    cz, cy, cx = za["chunks"]
    lab_depth = int(attrs.get("output_channels", 0) or 0)
    if "source_z_slice" in attrs:
        z0, z1 = attrs["source_z_slice"]
    else:                      # native volume used as-is: centred window of the label depth
        d0 = lab_depth or Z
        z0 = (Z - d0) // 2
        z1 = z0 + d0
    pool = 4 if (z1 - z0) == 4 * int(attrs.get("output_channels", 21)) else 1
    depth = (z1 - z0) // pool
    n = int(tiles.sum())
    chunk_mb = cz * cy * cx / 1e6
    info = dict(segment=seg, family=fam, scroll=seg.split("-")[0] if "-" in seg else "pherc0139", source=url,
                level=level, shape=[Z, H, W], z=[z0, z1], pool=pool, depth=depth, tiles=n,
                download_gb=round(n * chunk_mb / 1e3, 2), ink_px=int((ink > 0).sum()), sup_px=int((sup > 0).sum()),
                val_px=int((val > 0).sum()) if val is not None else 0, compressor=za["compressor"])
    if plan_only:
        return info
    assert cy == TILE and cx == TILE and cz == Z, f"unexpected source chunking {za['chunks']}"
    codec = numcodecs.get_codec(dict(za["compressor"])) if za["compressor"] else None
    sep = za.get("dimension_separator", ".")
    vol = zarr.open(os.path.join(d, "volume.zarr"), mode="a", shape=(depth, H, W), chunks=(depth, TILE, TILE),
                    dtype="u1", compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE), fill_value=0)
    done_path = os.path.join(d, "done_tiles.npy")
    done = np.load(done_path) if os.path.exists(done_path) else np.zeros_like(tiles)
    todo = [(iy, ix) for iy, ix in zip(*np.nonzero(tiles & ~done))]

    def one(k):
        iy, ix = k
        r = get(f"{url}/{level}/0{sep}{iy}{sep}{ix}")
        if r is None:
            return k
        raw = codec.decode(r.content) if codec else r.content
        a = np.frombuffer(raw, np.uint8).reshape(cz, cy, cx)[z0:z1].astype(np.float32)
        if pool > 1:
            a = a.reshape(depth, pool, cy, cx).mean(1)
        a = np.rint(a).astype(np.uint8)
        y0, x0 = iy * TILE, ix * TILE
        vol[:, y0:min(y0 + TILE, H), x0:min(x0 + TILE, W)] = a[:, :min(TILE, H - y0), :min(TILE, W - x0)]
        return k
    t0 = time.time()
    with ThreadPoolExecutor(24) as ex:
        for i, k in enumerate(ex.map(one, todo)):
            done[k] = True
            if i % 200 == 199:
                np.save(done_path, done)
                print(f"  {seg}: {i + 1}/{len(todo)} tiles, {(i + 1) * chunk_mb / (time.time() - t0):.1f} MB/s", flush=True)
    np.save(done_path, done)
    json.dump(info, open(os.path.join(d, "info.json"), "w"), indent=1)
    return info


if __name__ == "__main__":
    cmd = sys.argv[1]
    segs = segments()
    if cmd == "list":
        for fam, s in segs:
            print(fam, s)
    elif cmd in ("plan", "fetch"):
        want = set(sys.argv[2:])
        tot = 0
        # validation segments first, then one or two per scroll, so training can start before everything is here
        first = ["pherc0139-w016", "pherc0814-46527", "pherc1667-w029", "pherc0139-w040", "pherc1667-w013",
                 "phercparis4-w00", "phercparis4-w02", "pherc0139-w043", "pherc1667-w023"]
        segs = sorted(segs, key=lambda fs: first.index(fs[1]) if fs[1] in first else len(first))
        for fam, s in segs:
            if want and s not in want:
                continue
            try:
                info = fetch(fam, s, plan_only=(cmd == "plan"))
            except Exception as e:
                print(f"{s}: FAILED {str(e)[:200]}", flush=True)
                continue
            tot += info["download_gb"]
            print(f"{s:24s} {info['family'][:7]} src {info['shape']} z{info['z']} pool {info['pool']} tiles {info['tiles']:5d} "
                  f"dl {info['download_gb']:6.2f} GB  ink {info['ink_px']:>9d} sup {info['sup_px']:>9d} val {info['val_px']:>8d}", flush=True)
        print(f"TOTAL download {tot:.1f} GB")
