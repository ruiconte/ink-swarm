#!/bin/bash
# usage: docker run --gpus all -e ZARR_URL=... -v $PWD/mesh:/mesh -v $PWD/out:/out IMAGE [SIDE] [NAME]
#   /mesh = the submitted flattened tifxyz (x.tif, y.tif, z.tif, meta.json); SIDE = fwd or rev: the layer order in which the letters appear (look at both).
set -e; cd /app
SIDE=${1:-rev}; NAME=${2:-$(python -c "import json;print(json.load(open('/mesh/meta.json'))['uuid'])")}
URL=${ZARR_URL:?set ZARR_URL to the scan OME-Zarr, e.g. https://vesuvius-challenge-open-data.s3.amazonaws.com/<SCROLL>/volumes/<SCAN>.zarr}
W=/out/work; mkdir -p $W/mesh; cp /mesh/* $W/mesh/
[ -f $W/sv.npy ] || python -u herculex_port/render_tifxyz.py $W/mesh $URL $W/sv.npy 16      # 33 layers along the normals
FINAL_MODELS="argus-v3=release/argus-v3/argus-v3.pth" READ=fwd python -u hunt/final4.py $W/sv.npy $W   # Argus V3, 5 depths, both orders
python -u hunt/studio_read.py $W                                                                 # Studio, 3 depths, both orders
python -u make_claim.py $W $SIDE $NAME /out
