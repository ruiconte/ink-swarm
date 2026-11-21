#!/bin/bash
# Replays the Dockerfile on a machine already running its base image (used to test it on a rented GPU), then runs it.
set -e; cd /tmp/ctx
grep -E '^RUN pip' Dockerfile | sed 's/^RUN //' | bash
mkdir -p /app && cp -r code/. /app/ && cp run_claim.sh /app/run_claim.sh
grep -E '^ADD ' Dockerfile | while read _ url dst; do mkdir -p $(dirname $dst); python -c "import urllib.request,sys;urllib.request.urlretrieve(sys.argv[1],sys.argv[2])" "$url" $dst; done
cd /app && grep -oE '[0-9a-f]{64}  [^"]+' /tmp/ctx/Dockerfile > /app/SHA256 && sha256sum -c /app/SHA256
mkdir -p /mesh /out && cp /tmp/ctx/mesh_tifxyz/* /mesh/ && cd /app && bash run_claim.sh ${SIDE:-rev}
