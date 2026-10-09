"""Test-studio server: kev.serve backend + static UI, one runner.

    .venv/bin/python ui/server.py --run results/td-delta-v2-9b [--port 8010]
"""
from __future__ import annotations

import argparse
import os
import sys
import torch

from fastapi.staticfiles import StaticFiles

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "third_party", "kev"))

from kev.serve import (  # noqa: E402
    Checkpoint,
    LoadOptions,
    Server,
    app,
    default_device,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/td-delta-v2-9b")
    ap.add_argument("--port", type=int, default=8010)
    a = ap.parse_args()
    # Torch backend on Apple Silicon: the MLX path does not carry our LoRA head.
    os.environ.setdefault("KEV_BACKEND", "torch")
    dev = default_device()
    opts = LoadOptions.from_env()
    if dev == "mps" and opts.attn is None:
        from dataclasses import replace
        opts = replace(opts, attn="sdpa")
    if dev != "cpu" and opts.dtype is None:
        from dataclasses import replace
        opts = replace(opts, dtype=torch.bfloat16)
    ck = Checkpoint(a.run)
    tok, model = ck.load(dev, opts)
    app.state.server = Server(ck, tok, model, dev)
    here = os.path.dirname(os.path.abspath(__file__))
    app.mount("/", StaticFiles(directory=here, html=True), name="ui")
    print(f"studio: {ck.requested} on {dev} via {model.backend} -> http://127.0.0.1:{a.port}/")
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=a.port)


if __name__ == "__main__":
    main()
