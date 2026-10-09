# Kev RL post-training env (replicates the Vast setup byte-for-byte intent).
#
# Build (on a machine with docker, e.g. judith):
#   docker build -t <registry>/kev-train:cu128-py313 --build-arg KEV_REF=main docker/
#   docker push <registry>/kev-train:cu128-py313
# Use on Vast: launch instance with image `<registry>/kev-train:cu128-py313`.
# Use anywhere: `docker run --gpus all -it <registry>/kev-train:cu128-py313`
#
# Notes:
# - CUDA *runtime* image; the host supplies the driver (NVIDIA container toolkit).
# - Python comes from `uv` (3.13), matching kev's requires-python (>=3.12,<3.14).
# - `KEV_REF` pins the kev revision; default tracks main (pass a SHA for exactness).
# - Model weights, datasets and run outputs are NOT baked in (large, fast to
#   re-pull on a fat link); `inet_down` matters more than image size.

FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PATH="/root/.local/bin:$PATH"

RUN apt-get update -qq \
    && apt-get install -y -qq git curl rsync autossh aria2 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && uv python install 3.13

ARG KEV_REF=main

RUN git clone https://github.com/jaredpalmer/kev /root/kev \
    && cd /root/kev \
    && git checkout "${KEV_REF}" \
    && uv sync --extra serve --python 3.13 \
    && VIRTUAL_ENV=/root/kev/.venv uv pip install -q hf-xet hf_transfer

# Runtime convention (not baked in): HF_TOKEN via /root/.hf_token (mode 600),
# training data + runs scp'd per job, weights re-pulled on a fat link.

# SSH access: vast key injection is unreliable with custom images, so bake
# the operator PUBLIC key in (public keys are public; never bake private keys).
# Pass at build time: --build-arg OPERATOR_PUBKEY="$(cat ~/.ssh/vast_ed25519.pub)"
ARG OPERATOR_PUBKEY=""
RUN mkdir -p /root/.ssh && chmod 700 /root/.ssh \
    && if [ -n "$OPERATOR_PUBKEY" ]; then echo "$OPERATOR_PUBKEY" >> /root/.ssh/authorized_keys; fi \
    && chmod 600 /root/.ssh/authorized_keys || true

WORKDIR /root/kev

CMD ["bash"]
