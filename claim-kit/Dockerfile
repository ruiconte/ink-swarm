# First Letters claim kit: CT volume + flattened tifxyz mesh -> claim image + controls. Needs an NVIDIA GPU (>= 12 GB).
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
RUN pip install --no-cache-dir "numpy<2.3" scipy pillow tifffile "zarr>=2.18,<3" numcodecs fsspec aiohttp requests safetensors
WORKDIR /app
COPY code/ /app/
ADD https://huggingface.co/ruiconte/argus-v3/resolve/main/argus-v3.pth /app/release/argus-v3/argus-v3.pth
ADD https://github.com/priyankaranke/vesuvius-studio-multiscroll-cpt/releases/download/studio-softloss-20261023/studio-softloss-20261023-ema.safetensors /app/ext/studio/studio_softloss_ema.safetensors
COPY run_claim.sh /app/run_claim.sh
RUN echo "2d887d2bffaadcefae1507bea269e04e1e79c47a6320cce6071d701b4089bbe3  release/argus-v3/argus-v3.pth" > /app/SHA256 && \
    echo "96b326286e38de99047f7cab36170c53749a18d5f43e5fde0c7c20f2980b0e90  ext/studio/studio_softloss_ema.safetensors" >> /app/SHA256 && \
    sha256sum -c /app/SHA256
ENTRYPOINT ["bash", "/app/run_claim.sh"]
