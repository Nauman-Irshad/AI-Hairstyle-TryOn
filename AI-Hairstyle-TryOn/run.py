"""
Launch Uvicorn for the AI Hairstyle Try-On API + static UI.
Run from project root:  python run.py

If port 8000 is busy (another run.py still open), the next free port (8001, 8002, …) is used.
Override with:  set PORT=9000   (Windows PowerShell: $env:PORT=9000; python run.py)

Auto-reload: default ON (Linux/macOS), default OFF on Windows (set RELOAD=1 to enable).
Windows: listen host defaults to 127.0.0.1 (set HOST=0.0.0.0 for LAN). WinError 10013: use RELOAD=0 or unset RELOAD.
"""

import os
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import uvicorn


def _port_available(port: int, host: str) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
            return True
        except OSError:
            return False


def choose_port(host: str, default_start: int = 8000, max_attempts: int = 20) -> int:
    """
    Pick a listen port. If PORT is set, it must be bindable on `host` or we raise
    (avoids uvicorn failing later with WinError 10013 / address in use).
    """
    env = os.environ.get("PORT") or os.environ.get("UVICORN_PORT")
    if env:
        p = int(env.strip())
        if _port_available(p, host):
            return p
        raise RuntimeError(
            f"PORT={p} is not available on host {host!r} (in use, or blocked). "
            "Try another port (e.g. 18080) or close the process using it. "
            "On Windows, WinError 10013 on 0.0.0.0: set HOST=127.0.0.1"
        )
    for port in range(default_start, default_start + max_attempts):
        if _port_available(port, host):
            return port
    raise RuntimeError(f"No free TCP port in range {default_start}–{default_start + max_attempts - 1} on {host!r}")


def _reload_enabled() -> bool:
    # Uvicorn's file reloader on Windows often hits socket errors (e.g. WinError 10013); opt in with RELOAD=1.
    default = "0" if sys.platform == "win32" else "1"
    v = os.environ.get("RELOAD", default).strip().lower()
    return v not in ("0", "false", "no", "off")


if __name__ == "__main__":
    # Windows: binding 0.0.0.0 sometimes raises WinError 10013 (access denied); loopback is safer for local dev.
    default_host = "127.0.0.1" if sys.platform == "win32" else "0.0.0.0"
    host = os.environ.get("HOST", default_host)
    port = choose_port(host)
    env_port = os.environ.get("PORT") or os.environ.get("UVICORN_PORT")
    if not env_port and port != 8000:
        print(f"[run.py] Port 8000 was busy — using port {port} instead.", flush=True)
    elif env_port:
        print(f"[run.py] Using PORT={port} HOST={host}", flush=True)
    print(f"[run.py] Open: http://127.0.0.1:{port}/", flush=True)
    reload = _reload_enabled()
    if reload:
        print("[run.py] Auto-reload ON (set RELOAD=0 to disable).", flush=True)
    else:
        print("[run.py] Auto-reload OFF — restart manually after code changes.", flush=True)
    uvicorn_kwargs = {
        "host": host,
        "port": port,
        "reload": reload,
    }
    if reload:
        uvicorn_kwargs["reload_dirs"] = [str(ROOT)]
        uvicorn_kwargs["reload_excludes"] = [
            "**/outputs/**",
            "**/__pycache__/**",
            "**/.git/**",
            "**/models/**",
            "**/*.png",
            "**/*.jpg",
            "**/*.jpeg",
            "**/*.zip",
            "**/*.pth",
            "**/*.onnx",
        ]
    uvicorn.run("app.backend.main:app", **uvicorn_kwargs)
