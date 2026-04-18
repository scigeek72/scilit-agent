"""
services.py — Lifecycle management for optional external services.

Currently manages:
  - Grobid (Docker container on port 8070)

Nougat requires no daemon — it runs as a CLI subprocess per PDF and is
managed directly by NougatParser. No lifecycle code needed here.

Usage (called automatically by the app at startup):
    from services import ensure_grobid
    ensure_grobid()   # starts container if not already running; no-op if up

The user never needs to run docker commands manually. The app handles it.

Requirements:
  - Docker must be installed and the Docker daemon must be running.
  - The lfoppiano/grobid:0.8.0 image will be pulled automatically on first run.
"""

from __future__ import annotations

import logging
import subprocess
import time

import requests

from config import Config

logger = logging.getLogger(__name__)

# ── Grobid settings ──────────────────────────────────────────────────────────
_GROBID_IMAGE   = "lfoppiano/grobid:0.8.0"
_GROBID_CONTAINER_NAME = "scilit-grobid"
_GROBID_PORT    = 8070
_GROBID_HEALTH  = f"{{}}/api/isalive"   # format with base URL
_STARTUP_WAIT_S = 60    # max seconds to wait for Grobid to become healthy
_POLL_INTERVAL  = 2.0   # seconds between health-check polls


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def ensure_grobid(auto_start: bool = True) -> bool:
    """
    Ensure Grobid is available.

    1. If already responding → return True immediately.
    2. If not responding and auto_start=True → try to start the Docker container.
    3. If Docker is unavailable or container fails to start → return False
       (the parser router will fall back to PyMuPDF automatically).

    Returns True if Grobid is ready, False otherwise.
    """
    if _grobid_is_healthy():
        logger.info("Grobid already running at %s", Config.GROBID_URL)
        return True

    if not auto_start:
        logger.info("Grobid not running and auto_start=False — will use PyMuPDF fallback")
        return False

    if not _docker_available():
        logger.warning(
            "Docker not found. Install Docker Desktop to enable Grobid. "
            "Falling back to PyMuPDF."
        )
        return False

    # Check if a stopped container already exists and restart it
    if _container_exists(_GROBID_CONTAINER_NAME):
        logger.info("Restarting existing Grobid container '%s'", _GROBID_CONTAINER_NAME)
        _run(["docker", "start", _GROBID_CONTAINER_NAME])
    else:
        logger.info("Starting Grobid container (image: %s) ...", _GROBID_IMAGE)
        ok = _start_grobid_container()
        if not ok:
            return False

    return _wait_for_grobid()


def stop_grobid() -> None:
    """
    Stop the Grobid container (graceful shutdown).
    Safe to call even if the container is not running.
    """
    if _container_exists(_GROBID_CONTAINER_NAME):
        logger.info("Stopping Grobid container '%s'", _GROBID_CONTAINER_NAME)
        _run(["docker", "stop", _GROBID_CONTAINER_NAME])
    else:
        logger.debug("Grobid container not found — nothing to stop")


def grobid_status() -> dict:
    """Return a status dict for display in the UI or CLI."""
    healthy = _grobid_is_healthy()
    docker_ok = _docker_available()
    container_running = _container_is_running(_GROBID_CONTAINER_NAME)
    return {
        "healthy": healthy,
        "docker_available": docker_ok,
        "container_running": container_running,
        "url": Config.GROBID_URL,
        "image": _GROBID_IMAGE,
    }


def marker_status() -> dict:
    """Return a status dict for marker (pip-installed CLI, no daemon)."""
    available = _marker_installed()
    return {
        "available": available,
        "install_cmd": "pip install marker-pdf",
        "note": "No daemon required — runs per-PDF as a subprocess.",
    }


# Keep backward-compatible alias
def nougat_status() -> dict:
    """Deprecated alias for marker_status()."""
    return marker_status()


def print_service_status() -> None:
    """Print a human-readable service status table to stdout."""
    g = grobid_status()
    m = marker_status()
    lines = [
        "",
        "  Service Status",
        "  ─────────────────────────────────────────────────────",
        f"  Grobid  : {'✓ running' if g['healthy'] else '✗ not running'}",
        f"            Docker available : {g['docker_available']}",
        f"            Container exists : {g['container_running']}",
        f"            URL              : {g['url']}",
        f"  marker  : {'✓ installed' if m['available'] else '✗ not installed'}",
        f"            {m['note']}",
        f"            Install          : {m['install_cmd']}",
        "  PyMuPDF : ✓ always available (fallback)",
        "  ─────────────────────────────────────────────────────",
        "",
    ]
    print("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _grobid_is_healthy() -> bool:
    """Return True if Grobid's health endpoint responds 200."""
    try:
        url = Config.GROBID_URL.rstrip("/") + "/api/isalive"
        resp = requests.get(url, timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


def _docker_available() -> bool:
    """Return True if the docker CLI is on PATH and the daemon is running."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _container_exists(name: str) -> bool:
    """Return True if a container with this name exists (running or stopped)."""
    try:
        result = subprocess.run(
            ["docker", "ps", "-a", "--filter", f"name=^{name}$", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return name in result.stdout
    except Exception:
        return False


def _container_is_running(name: str) -> bool:
    """Return True if a container with this name is currently running."""
    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", f"name=^{name}$", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return name in result.stdout
    except Exception:
        return False


def _start_grobid_container() -> bool:
    """
    Start a new Grobid container (pulls image on first run).
    Returns True if the docker run command succeeded.
    """
    cmd = [
        "docker", "run",
        "--name", _GROBID_CONTAINER_NAME,
        "--detach",
        "--restart", "unless-stopped",   # survives Docker Desktop restarts
        "-p", f"{_GROBID_PORT}:{_GROBID_PORT}",
        _GROBID_IMAGE,
    ]
    result = _run(cmd)
    if result is None or result.returncode != 0:
        stderr = result.stderr if result else "docker run failed"
        logger.error("Failed to start Grobid container: %s", stderr)
        return False
    return True


def _wait_for_grobid() -> bool:
    """
    Poll Grobid's health endpoint until it responds or timeout is reached.
    Returns True if it becomes healthy within _STARTUP_WAIT_S seconds.
    """
    logger.info(
        "Waiting up to %ds for Grobid to become healthy ...", _STARTUP_WAIT_S
    )
    deadline = time.time() + _STARTUP_WAIT_S
    while time.time() < deadline:
        if _grobid_is_healthy():
            logger.info("Grobid is healthy at %s", Config.GROBID_URL)
            return True
        time.sleep(_POLL_INTERVAL)

    logger.warning(
        "Grobid did not become healthy within %ds. "
        "Falling back to PyMuPDF.",
        _STARTUP_WAIT_S,
    )
    return False


def _marker_installed() -> bool:
    """Return True if the marker_single CLI is available on PATH."""
    try:
        result = subprocess.run(
            ["marker_single", "--help"],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _nougat_installed() -> bool:
    """Deprecated — checks marker_single instead."""
    return _marker_installed()


def _run(cmd: list[str]) -> subprocess.CompletedProcess | None:
    """Run a shell command, return CompletedProcess or None on error."""
    try:
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception as exc:
        logger.debug("Command failed %s: %s", cmd, exc)
        return None
