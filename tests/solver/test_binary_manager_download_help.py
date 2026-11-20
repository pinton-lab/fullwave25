"""Tests for cached-binary permissions and download-failure messages."""

import logging
import re
import ssl
import urllib.error
from pathlib import Path

import pytest

from fullwave.solver import binary_manager

TAG = "test_tag"
FILENAME = "fullwave_solver_test"


@pytest.fixture
def cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(binary_manager, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setenv("FULLWAVE25_BINARY_TAG", TAG)
    return tmp_path / "cache"


@pytest.fixture
def missing_bundled_path(tmp_path: Path) -> Path:
    return tmp_path / "bundled" / FILENAME


def _fail_download_with(*, monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    def raise_error(url: str, dest: Path) -> None:
        raise error

    monkeypatch.setattr(binary_manager, "_download_binary", raise_error)


def test_cached_binary_without_execute_bit_is_made_executable(
    cache_dir: Path,
    missing_bundled_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    cached = cache_dir / TAG / FILENAME
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"binary")
    cached.chmod(0o644)

    result = binary_manager.ensure_binary(missing_bundled_path)

    assert result == cached
    assert cached.stat().st_mode & 0o777 == 0o744
    assert str(cached) in caplog.text


def test_executable_cached_binary_is_returned_unchanged(
    cache_dir: Path,
    missing_bundled_path: Path,
) -> None:
    cached = cache_dir / TAG / FILENAME
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"binary")
    cached.chmod(0o755)

    result = binary_manager.ensure_binary(missing_bundled_path)

    assert result == cached
    assert cached.stat().st_mode & 0o777 == 0o755


def test_failed_chmod_tells_user_to_run_chmod(
    cache_dir: Path,
    missing_bundled_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cached = cache_dir / TAG / FILENAME
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"binary")
    cached.chmod(0o644)

    def refuse_chmod(self: Path, mode: int) -> None:
        raise PermissionError

    monkeypatch.setattr(Path, "chmod", refuse_chmod)

    with pytest.raises(RuntimeError, match=re.escape(f"chmod +x {cached}")):
        binary_manager.ensure_binary(missing_bundled_path)


@pytest.mark.parametrize(
    "error",
    [
        urllib.error.URLError(ssl.SSLCertVerificationError("self-signed certificate")),
        ssl.SSLCertVerificationError("self-signed certificate"),
    ],
    ids=["wrapped_in_url_error", "bare"],
)
def test_certificate_failure_message_explains_manual_placement_and_certificate(
    error: Exception,
    cache_dir: Path,
    missing_bundled_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fail_download_with(monkeypatch=monkeypatch, error=error)

    with pytest.raises(RuntimeError) as exc_info:
        binary_manager.ensure_binary(missing_bundled_path)

    message = str(exc_info.value)
    cached = cache_dir / TAG / FILENAME
    assert binary_manager._download_url(FILENAME, TAG) in message
    assert str(cached.parent) in message
    assert f"chmod +x {cached}" in message
    assert "SSL_CERT_FILE" in message


def test_connection_failure_message_has_no_certificate_line(
    cache_dir: Path,
    missing_bundled_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fail_download_with(
        monkeypatch=monkeypatch,
        error=urllib.error.URLError(ConnectionRefusedError()),
    )

    with pytest.raises(RuntimeError) as exc_info:
        binary_manager.ensure_binary(missing_bundled_path)

    message = str(exc_info.value)
    assert f"chmod +x {cache_dir / TAG / FILENAME}" in message
    assert "SSL_CERT_FILE" not in message


def test_http_404_message_has_manual_steps_without_certificate_line(
    cache_dir: Path,
    missing_bundled_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = urllib.error.HTTPError("https://example.invalid", 404, "Not Found", {}, None)
    _fail_download_with(monkeypatch=monkeypatch, error=error)

    with pytest.raises(RuntimeError) as exc_info:
        binary_manager.ensure_binary(missing_bundled_path)

    message = str(exc_info.value)
    cached = cache_dir / TAG / FILENAME
    assert binary_manager._download_url(FILENAME, TAG) in message
    assert str(cached.parent) in message
    assert f"chmod +x {cached}" in message
    assert "SSL_CERT_FILE" not in message
