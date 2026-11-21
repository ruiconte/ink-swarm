# claim-kit: distance-true surfaces and automatic First Letters claim packages

Two tools, MIT licence. Start from a HerculeX (or any tifxyz) surface on a public scan, and end with the images and
files a First Letters submission asks for. Everything is programmatic, with no manual annotation.

## 1. `flatten_mesh.py`: isometric flattening of a tifxyz
```
python flatten_mesh.py MESH_IN 16 MESH_OUT
```
- Moves the 2D grid positions so that every grid edge and diagonal keeps its 3D length (Adam on GPU, 3000 steps).
- Then resamples the 3D points on a regular grid of the new coordinates and writes a new tifxyz (x/y/z.tif, meta.json).
- Any render of `MESH_OUT` is distance-true: 1 px = 1 voxel in both directions, so a scale bar is exact.
- Before/after distortion statistics are stored in `meta.json["flattening"]`.
- Example (22 × 22 mm HerculeX piece): grid steps within 10 % of their true length went from 86.5 % to 90.3 %; median step 0.997.

## 2. `make_claim.py` + `Dockerfile`: the claim package
```
docker build -t claimkit .
docker run --gpus all -e ZARR_URL=https://.../<SCAN>.zarr -v $PWD/mesh:/mesh -v $PWD/out:/out claimkit rev
```
1. **Render.** 33 layers along the normals from the public OME-Zarr (`herculex_port/render_tifxyz.py`), with the
   sheet followed pixel by pixel.
2. **Argus V3.** 17 layers, 5 depths, read in both layer orders (`hunt/final4.py`).
3. **Studio** (CC-BY-NC weights by priyankaranke / paranke). 16 layers, 3 depths, both orders (`hunt/studio_read.py`).
4. **`make_claim.py`.** Picks the 2 × 2 cm square where letter-side ink most exceeds control ink, then writes:
   - `NAME.png`: the claim image, with a 1 cm scale bar, line numbers in the margin, and three automatically measured letter-sized marks (px and mm);
   - `NAME_controls.png`: the same square, both models, letter side vs reversed layer order;
   - `NAME_whole_surface.png` and `NAME_claim.json`.

`SIDE` (`fwd` / `rev`) is the layer order in which the letters appear. **Look at both.** The side with more ink is not
always the side with the letters; the difference between the two is shape and line structure, not quantity.

The weights are downloaded from their public sources and checked by SHA-256 during the build.
`docker_replay.sh` replays the Dockerfile on a machine already running its base image, which is handy on rented GPUs
where Docker-in-Docker is not available. That path was tested end to end on an RTX 5080.

## Credits
HerculeX (surfaces), Studio (priyankaranke / paranke), Reader v3 (domenicor046), the Vesuvius Challenge team
(`ink_9um`, open data).
