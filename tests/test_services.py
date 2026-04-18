"""
tests/test_services.py — Unit tests for services.py (Grobid lifecycle).

All Docker calls are mocked — no real Docker daemon or container needed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestGrobidStatus:

    @patch("services.requests.get")
    def test_grobid_is_healthy_true(self, mock_get):
        from services import _grobid_is_healthy
        mock_get.return_value = MagicMock(status_code=200)
        assert _grobid_is_healthy() is True

    @patch("services.requests.get")
    def test_grobid_is_healthy_false_non_200(self, mock_get):
        from services import _grobid_is_healthy
        mock_get.return_value = MagicMock(status_code=503)
        assert _grobid_is_healthy() is False

    @patch("services.requests.get", side_effect=Exception("refused"))
    def test_grobid_is_healthy_false_on_exception(self, mock_get):
        from services import _grobid_is_healthy
        assert _grobid_is_healthy() is False


class TestDockerHelpers:

    @patch("services.subprocess.run")
    def test_docker_available_true(self, mock_run):
        from services import _docker_available
        mock_run.return_value = MagicMock(returncode=0)
        assert _docker_available() is True

    @patch("services.subprocess.run")
    def test_docker_available_false_nonzero(self, mock_run):
        from services import _docker_available
        mock_run.return_value = MagicMock(returncode=1)
        assert _docker_available() is False

    @patch("services.subprocess.run", side_effect=FileNotFoundError)
    def test_docker_available_false_not_found(self, mock_run):
        from services import _docker_available
        assert _docker_available() is False

    @patch("services.subprocess.run")
    def test_container_exists_true(self, mock_run):
        from services import _container_exists
        mock_run.return_value = MagicMock(returncode=0, stdout="scilit-grobid\n")
        assert _container_exists("scilit-grobid") is True

    @patch("services.subprocess.run")
    def test_container_exists_false(self, mock_run):
        from services import _container_exists
        mock_run.return_value = MagicMock(returncode=0, stdout="")
        assert _container_exists("scilit-grobid") is False

    @patch("services.subprocess.run")
    def test_container_is_running_true(self, mock_run):
        from services import _container_is_running
        mock_run.return_value = MagicMock(returncode=0, stdout="scilit-grobid\n")
        assert _container_is_running("scilit-grobid") is True


class TestEnsureGrobid:

    @patch("services._grobid_is_healthy", return_value=True)
    def test_already_healthy_returns_true_immediately(self, mock_health):
        from services import ensure_grobid
        result = ensure_grobid()
        assert result is True
        mock_health.assert_called_once()

    @patch("services._grobid_is_healthy", return_value=False)
    def test_auto_start_false_returns_false(self, mock_health):
        from services import ensure_grobid
        result = ensure_grobid(auto_start=False)
        assert result is False

    @patch("services._grobid_is_healthy", return_value=False)
    @patch("services._docker_available", return_value=False)
    def test_no_docker_returns_false(self, mock_docker, mock_health):
        from services import ensure_grobid
        result = ensure_grobid(auto_start=True)
        assert result is False

    @patch("services._grobid_is_healthy", return_value=False)
    @patch("services._docker_available", return_value=True)
    @patch("services._container_exists", return_value=True)
    @patch("services._run")
    @patch("services._wait_for_grobid", return_value=True)
    def test_restarts_existing_stopped_container(
        self, mock_wait, mock_run, mock_exists, mock_docker, mock_health
    ):
        from services import ensure_grobid
        result = ensure_grobid(auto_start=True)
        assert result is True
        # Should call docker start, not docker run
        mock_run.assert_called_once()
        assert mock_run.call_args[0][0][1] == "start"

    @patch("services._grobid_is_healthy", return_value=False)
    @patch("services._docker_available", return_value=True)
    @patch("services._container_exists", return_value=False)
    @patch("services._start_grobid_container", return_value=True)
    @patch("services._wait_for_grobid", return_value=True)
    def test_starts_new_container_when_none_exists(
        self, mock_wait, mock_start, mock_exists, mock_docker, mock_health
    ):
        from services import ensure_grobid
        result = ensure_grobid(auto_start=True)
        assert result is True
        mock_start.assert_called_once()

    @patch("services._grobid_is_healthy", return_value=False)
    @patch("services._docker_available", return_value=True)
    @patch("services._container_exists", return_value=False)
    @patch("services._start_grobid_container", return_value=False)
    def test_container_start_failure_returns_false(
        self, mock_start, mock_exists, mock_docker, mock_health
    ):
        from services import ensure_grobid
        result = ensure_grobid(auto_start=True)
        assert result is False


class TestWaitForGrobid:

    @patch("services._grobid_is_healthy", return_value=True)
    @patch("services.time.sleep")
    def test_healthy_immediately_returns_true(self, mock_sleep, mock_health):
        from services import _wait_for_grobid
        result = _wait_for_grobid()
        assert result is True
        mock_sleep.assert_not_called()

    @patch("services._grobid_is_healthy", side_effect=[False, False, True])
    @patch("services.time.sleep")
    def test_becomes_healthy_after_polls(self, mock_sleep, mock_health):
        from services import _wait_for_grobid
        result = _wait_for_grobid()
        assert result is True


class TestMarkerStatus:

    @patch("services.subprocess.run")
    def test_marker_installed_true(self, mock_run):
        from services import _marker_installed
        mock_run.return_value = MagicMock(returncode=0)
        assert _marker_installed() is True

    @patch("services.subprocess.run", side_effect=FileNotFoundError)
    def test_marker_installed_false(self, mock_run):
        from services import _marker_installed
        assert _marker_installed() is False


class TestStopGrobid:

    @patch("services._container_exists", return_value=True)
    @patch("services._run")
    def test_stop_calls_docker_stop(self, mock_run, mock_exists):
        from services import stop_grobid
        stop_grobid()
        mock_run.assert_called_once()
        assert mock_run.call_args[0][0][1] == "stop"

    @patch("services._container_exists", return_value=False)
    @patch("services._run")
    def test_stop_is_noop_when_no_container(self, mock_run, mock_exists):
        from services import stop_grobid
        stop_grobid()
        mock_run.assert_not_called()
