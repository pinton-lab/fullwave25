import logging

import pytest

from fullwave.solver import cuda_utils
from fullwave.utils.gpu_memory import free_gpu_memory_bytes


def test_free_gpu_memory_bytes_reads_each_cuda_ordinal(monkeypatch):
    asked = []

    def fake_free_bytes(*, ordinal):
        asked.append(ordinal)
        return 1000 + ordinal

    monkeypatch.setattr(cuda_utils, "free_memory_bytes_of_cuda_ordinal", fake_free_bytes)
    assert free_gpu_memory_bytes(["1", " 0"]) == {"1": 1001, "0": 1000}
    assert asked == [1, 0]


def test_free_gpu_memory_bytes_is_none_when_the_driver_fails(monkeypatch):
    def failing_free_bytes(*, ordinal):
        message = f"Cannot open a CUDA context on GPU {ordinal}."
        raise RuntimeError(message)

    monkeypatch.setattr(cuda_utils, "free_memory_bytes_of_cuda_ordinal", failing_free_bytes)
    assert free_gpu_memory_bytes(["0"]) is None


def test_free_gpu_memory_bytes_is_none_without_an_error_log_for_an_absent_gpu(
    monkeypatch,
    caplog,
):
    monkeypatch.setattr(cuda_utils, "count_visible_gpus", lambda: 1)
    with caplog.at_level(logging.WARNING):
        assert free_gpu_memory_bytes(["1"]) is None
    assert caplog.records == []


@pytest.mark.parametrize("device_ids", [["GPU-aaaa"], ["0,1"]])
def test_free_gpu_memory_bytes_is_none_for_an_id_that_is_not_an_ordinal(device_ids):
    assert free_gpu_memory_bytes(device_ids) is None
