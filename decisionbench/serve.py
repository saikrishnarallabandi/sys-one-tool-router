"""Start local System One servers (Kev, laya-serve) and wait until they answer.

    python -m decisionbench.serve kev-4b          # start, wait for health, serve until Ctrl-C
    python -m decisionbench.serve --list

The runner does the same automatically for models registered with `server=...`: it reuses a
server that is already healthy on the port (and leaves it running), otherwise it launches one and
stops it when the model is closed. Launching never installs anything: the server's environment
must already exist (see each spec's `setup`).
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from decisionbench.types import ModelUnavailable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KEV_DIR = PROJECT_ROOT / "third_party" / "kev"
KEV_PYTHON = KEV_DIR / ".venv" / "bin" / "python"
LOG_DIR = PROJECT_ROOT / ".cache" / "decisionbench" / "servers"
KEV_SETUP = ("(cd third_party/kev && uv sync --extra serve), after cloning https://github.com/jaredpalmer/kev "
             "into third_party/kev if it is not there (not `pip install kev`: that PyPI name is an unrelated package)")


@dataclass(frozen=True)
class ServerSpec:
    name: str
    cmd: tuple[str, ...]
    port: int
    health_path: str
    cwd: Path = PROJECT_ROOT
    env: dict = field(default_factory=dict)
    # Paths that must exist before launching, and how to create them.
    requires: tuple[Path, ...] = ()
    setup: str = ""
    # First launch downloads weights, so allow a long wait.
    ready_timeout_s: float = 1800.0

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


def _kev(name: str, run: str, port: int) -> ServerSpec:
    # The Kev venv's interpreter is called directly rather than `uv run`, which would sync (install) first.
    return ServerSpec(name, (str(KEV_PYTHON), "-m", "kev.serve", "--run", run, "--port", str(port)), port,
                      "/v1/models", cwd=KEV_DIR, requires=(KEV_PYTHON,), setup=KEV_SETUP)


SERVERS: dict[str, ServerSpec] = {
    "kev-0.8b": _kev("kev-0.8b", "jaredpalmer/kev-0.8b", 8007),
    "kev-4b": _kev("kev-4b", "jaredpalmer/kev-4b", 8008),
    "kev-9b": _kev("kev-9b", "jaredpalmer/kev-9b", 8009),
    "laya-serve": ServerSpec(
        "laya-serve", (str(Path(sys.prefix) / "bin" / "laya-serve"),), 8000, "/health",
        env={"LAYA_HOST": "127.0.0.1", "LAYA_PORT": "8000", "LAYA_MODELS": "english", "LAYA_PRELOAD": "1"},
        requires=(Path(sys.prefix) / "bin" / "laya-serve",),
        setup="uv pip install --python .venv/bin/python 'laya[serve]'", ready_timeout_s=600.0),
}


def get_json(url: str, api_key: str | None = None, timeout: float = 5.0) -> dict | list | None:
    """GET a JSON document; None when the server is not reachable or answers non-200."""
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"} if api_key else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8")) if resp.status == 200 else None
    except (urllib.error.URLError, ConnectionError, TimeoutError, json.JSONDecodeError):
        return None


class ManagedServer:
    """A server process this harness may own. `start()` reuses a healthy server already on the port."""

    def __init__(self, spec: ServerSpec, api_key: str | None = "local"):
        self.spec, self.api_key = spec, api_key
        self.proc: subprocess.Popen | None = None
        self.owned = False
        self.log_path = LOG_DIR / f"{spec.name}.log"

    @property
    def base_url(self) -> str:
        return self.spec.base_url

    def healthy(self) -> bool:
        return get_json(self.base_url + self.spec.health_path, self.api_key) is not None

    def start(self) -> "ManagedServer":
        if self.healthy():
            return self
        missing = [str(p) for p in self.spec.requires if not p.exists()]
        if missing:
            raise ModelUnavailable(f"{self.spec.name} server is not installed ({', '.join(missing)} missing); "
                                   f"set it up with: {self.spec.setup}")
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log = self.log_path.open("ab")
        self.proc = subprocess.Popen(self.spec.cmd, cwd=self.spec.cwd, env={**os.environ, **self.spec.env},
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        log.close()
        self.owned = True
        atexit.register(self.stop)
        deadline = time.monotonic() + self.spec.ready_timeout_s
        while time.monotonic() < deadline:
            code = self.proc.poll()
            if code is not None:
                self.proc = None
                raise ModelUnavailable(f"{self.spec.name} server exited with code {code} before becoming ready; "
                                       f"log tail:\n{self.log_tail()}")
            if self.healthy():
                return self
            time.sleep(1.0)
        self.stop()
        raise ModelUnavailable(f"{self.spec.name} server not ready after {self.spec.ready_timeout_s:.0f}s; "
                               f"log tail:\n{self.log_tail()}")

    def stop(self) -> None:
        if not self.owned or self.proc is None:
            return
        proc, self.proc = self.proc, None
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()

    def log_tail(self, n: int = 15) -> str:
        if not self.log_path.exists():
            return "(no log)"
        return "\n".join(self.log_path.read_text(errors="replace").splitlines()[-n:])


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m decisionbench.serve", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", nargs="?", help="server to start: " + ", ".join(SERVERS))
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args(argv)
    if args.list or not args.name:
        for s in SERVERS.values():
            print(f"{s.name:12s} {s.base_url}  {' '.join(s.cmd)}")
        return
    if args.name not in SERVERS:
        ap.error(f"unknown server {args.name!r}; known: {', '.join(SERVERS)}")
    server = ManagedServer(SERVERS[args.name])
    try:
        server.start()
    except ModelUnavailable as e:
        sys.exit(str(e))
    info = get_json(server.base_url + server.spec.health_path, server.api_key)
    if not server.owned:
        print(f"{args.name} already running at {server.base_url}: {json.dumps(info)[:300]}")
        return
    print(f"{args.name} ready at {server.base_url} (log: {server.log_path}): {json.dumps(info)[:300]}\nCtrl-C to stop.")
    try:
        while server.proc is not None and server.proc.poll() is None:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == "__main__":
    main()
