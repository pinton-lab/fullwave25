from pathlib import Path

import pytest

from fullwave.utils import memory_tempfile


def test_run_directory_in_memory_uses_tmpfs_when_one_exists():
    probe = memory_tempfile.MemoryTempfile(filesystem_types=["tmpfs"], fallback=True)
    if not probe.found_mem_tempdir():
        pytest.skip("no tmpfs on this system")

    directory = memory_tempfile.run_directory_in_memory()
    try:
        assert Path(directory.name).is_dir()
        assert str(Path(directory.name).parent) in probe.get_usable_mem_tempdir_paths()
    finally:
        directory.cleanup()


def test_run_directory_in_memory_falls_back_without_tmpfs(monkeypatch, tmp_path):
    monkeypatch.setattr(memory_tempfile.tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(memory_tempfile, "SUITABLE_PATHS", [])
    real_init = memory_tempfile.MemoryTempfile.__init__

    def init_without_tmpfs(self, **kwargs):
        kwargs.update(filesystem_types=["no-such-filesystem"], preferred_paths=[])
        kwargs.update(additional_paths=[])
        real_init(self, **kwargs)

    monkeypatch.setattr(memory_tempfile.MemoryTempfile, "__init__", init_without_tmpfs)

    directory = memory_tempfile.run_directory_in_memory()
    try:
        assert Path(directory.name).parent == tmp_path
    finally:
        directory.cleanup()
