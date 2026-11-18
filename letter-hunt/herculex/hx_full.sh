#!/bin/bash
# Fresh box: install HerculeX upstream (Linux/CUDA), cut the scroll $SCROLL into pieces, then read every piece with
# reader-v3 FIRST (second opinion over the whole scroll), then Argus V3 (whole scroll). After each read the faces are
# packed into /workspace/faces_<model>.tgz and a line "=== PACKED <model>" goes to hx.log.
S=${SCROLL:?set SCROLL, e.g. SCROLL=PHerc0826}
log() { echo "=== $* $(date)" >> /workspace/hx.log; }
C=${CORES:-32}
export HERCULEX_BOARD_CORES=$C OMP_NUM_THREADS=$C HERCULEX_SAMPLERS=$((C*3/4)) HERCULEX_AHEAD=$((C-4))
cd /workspace
pip install -q uv
[ -d HerculeX ] || git clone -q https://github.com/adambetka2/HerculeX.git
cd HerculeX && git log --oneline -1 >> /workspace/hx.log
UV_PROJECT_ENVIRONMENT=runtime/python uv sync --locked --extra ink >> /workspace/hx.log 2>&1 || { log SYNC_FAILED; exit 1; }
./run.sh ink.model.weights fetch >> /workspace/hx.log 2>&1
sed -i 's/^PIECES_AHEAD = .*/PIECES_AHEAD = int(os.environ.get("HERCULEX_AHEAD", "6"))/' ink/reading/piece_reading.py
grep -q "^import os" ink/reading/piece_reading.py || sed -i '1i import os' ink/reading/piece_reading.py
grep -n "PIECES_AHEAD =" ink/reading/piece_reading.py >> /workspace/hx.log
log installed
(./run.sh runner.pipeline run sheets $S > /workspace/sheets.log 2>&1; echo "=== sheets rc $? $(date)" >> /workspace/hx.log) &
mkdir -p /workspace/weights
runtime/python/bin/python - >> /workspace/hx.log 2>&1 <<'PY'
import urllib.request as u, torch
for n, url in (("argus-v3", "https://huggingface.co/ruiconte/argus-v3/resolve/main/argus-v3.pth"),
               ("reader-v3", "https://huggingface.co/domenicor046/reader-v3/resolve/main/reader-v3.pth")):
    u.urlretrieve(url, f"/workspace/weights/{n}.pth")
    sd = torch.load(f"/workspace/weights/{n}.pth", map_location="cpu", weights_only=False)["model"]
    torch.save({"model." + k: v for k, v in sd.items()}, f"/workspace/weights/{n}-raw.pth"); print(n, len(sd))
PY
log "weights $?"
wait
./run.sh runner.pipeline run pieces $S > /workspace/pieces.log 2>&1; log "pieces rc $?"
export HERCULEX_KEEP_STACKS=0
F=/workspace/HerculeX/data/outputs/unwrapping/level/$S
for M in reader-v3 argus-v3; do
  export HERCULEX_WEIGHTS=/workspace/weights/$M-raw.pth HERCULEX_MODEL_NAME="$M"
  ./run.sh runner.pipeline run sheet $S > /workspace/sheet_$M.log 2>&1 || ./run.sh runner.pipeline run sheet $S >> /workspace/sheet_$M.log 2>&1
  log "read $M rc $? failed=$(grep -c failed /workspace/sheet_$M.log)"
  FACES=$(find $F -type d -name faces | head -1)
  (cd $FACES && find . -name '*-taught*' | tar czf /workspace/faces_$M.tgz -T -)
  find $FACES -name '*-taught*' -delete
  log "PACKED $M"
done
log ALL_DONE
