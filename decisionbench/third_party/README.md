# third_party

Upstream repos this project uses. They are not vendored; clone them at the pinned commits:

```bash
git clone https://github.com/jaredpalmer/kev third_party/kev
git -C third_party/kev checkout 569ea449b1033c5aa8f149ec5e20d0c7a296d24a
(cd third_party/kev && uv sync --extra serve)

git clone https://github.com/vinnylarouge/jevlike third_party/jevlike
git -C third_party/jevlike checkout 94f5fd1b0b11d52bbdfdf4e0ee6aa96b568f8452
```

- `kev`: training (`kev.train`), held-out evaluation (`kev.benchmark`), temperature fitting (`kev.metrics`)
  and serving (`kev.serve`, used by `ui/server.py`). The training runs in `notes/td-delta-run-log.md` used
  kev @ `ce089c6` on the GPU boxes; the local checkout above is the one the eval/serve code ran against.
- `jevlike`: the TinyScorer baseline (`scripts/jevlike_td_data.py`, `runs/jevlike-td.pt`).
