"""Binary manager for fullwave solver executables.

Handles locating, caching, and downloading solver binaries from GitHub releases.

Priority order when resolving a binary:
1. Bundled binary (shipped with the package or dev install).
2. Local cache at ``~/.cache/fullwave25/bins/``.
3. Download from the GitHub release matching the current package version.

The release tag can be overridden with the environment variable
``FULLWAVE25_BINARY_TAG`` (e.g. ``export FULLWAVE25_BINARY_TAG=v1.2.3``).
"""

import logging
import os
import stat
import urllib.error
import urllib.request
from pathlib import Path

logger = logging.getLogger("__main__." + __name__)

GITHUB_REPO = "pinton-lab/fullwave25"
CACHE_DIR = Path.home() / ".cache" / "fullwave25" / "bins"


def _release_tag() -> str:
    """Derive the GitHub release tag from the installed package version.

    Strips pre-release / dev suffixes so that ``1.2.2-dev0`` → ``v1.2.2``.
    """
    import fullwave  # local import to avoid circular at module load

    raw = fullwave.__version__
    # Strip common pre-release markers: -devN, aN, bN, rcN
    for sep in ("-", "a", "b", "rc"):
        raw = raw.split(sep)[0]
    return f"v{raw}"


def _download_url(filename: str, tag: str) -> str:
    return f"https://github.com/{GITHUB_REPO}/releases/download/{tag}/{filename}"


def _download_binary(url: str, dest: Path) -> None:
    """Download *url* to *dest*, showing a tqdm progress bar when available."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".download_tmp")
    try:
        try:
            from tqdm import tqdm

            class _Hook:
                def __init__(self) -> None:
                    self._bar: tqdm | None = None

                def __call__(self, block: int, block_size: int, total: int) -> None:  # noqa: ARG002
                    if self._bar is None:
                        self._bar = tqdm(
                            total=total if total > 0 else None,
                            unit="B",
                            unit_scale=True,
                            desc=dest.name,
                        )
                    self._bar.update(block_size)

                def close(self) -> None:
                    if self._bar is not None:
                        self._bar.close()

            hook = _Hook()
            try:
                urllib.request.urlretrieve(url, tmp, reporthook=hook)  # noqa: S310
            finally:
                hook.close()

        except ImportError:
            urllib.request.urlretrieve(url, tmp)  # noqa: S310

        tmp.rename(dest)
        dest.chmod(dest.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        logger.info("Binary saved to %s", dest)

    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def ensure_binary(local_path: Path) -> Path:
    """Return an executable path for the solver binary.

    Parameters
    ----------
    local_path:
        The expected bundled path (as computed by ``_retrieve_fullwave_simulation_path``).

    Returns
    -------
    Path
        Resolved path to an executable binary.

    Raises
    ------
    RuntimeError
        If the binary cannot be found locally and the download fails.

    """
    # 1. Bundled binary present (dev install or package that still ships binaries).
    if local_path.exists():
        logger.debug("Using bundled binary: %s", local_path)
        return local_path

    filename = local_path.name
    cached = CACHE_DIR / filename

    # 2. Previously downloaded and cached.
    if cached.exists():
        logger.debug("Using cached binary: %s", cached)
        return cached

    # 3. Download from GitHub releases.
    tag = os.environ.get("FULLWAVE25_BINARY_TAG", _release_tag())
    url = _download_url(filename, tag)

    logger.info(
        "Binary '%s' not found locally. Downloading from GitHub release '%s' …",
        filename,
        tag,
    )

    try:
        _download_binary(url, cached)
    except urllib.error.HTTPError as e:
        msg = (
            f"Could not download binary '{filename}' from:\n  {url}\n"
            f"HTTP error: {e.code} {e.reason}\n\n"
            "Possible fixes:\n"
            f"  • Check that release '{tag}' exists and contains '{filename}'.\n"
            f"  • Override the tag:  export FULLWAVE25_BINARY_TAG=<tag>\n"
            f"  • Place the binary manually at:  {cached}\n"
        )
        raise RuntimeError(msg) from e
    except Exception as e:
        msg = (
            f"Failed to download binary '{filename}' from:\n  {url}\n"
            f"Error: {e}\n\n"
            f"  • Place the binary manually at:  {cached}\n"
            f"  • Or override the tag:  export FULLWAVE25_BINARY_TAG=<tag>\n"
        )
        raise RuntimeError(msg) from e

    return cached
