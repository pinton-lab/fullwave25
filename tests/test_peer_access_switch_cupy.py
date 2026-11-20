from types import SimpleNamespace

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from fullwave.medium import _upload_or_convert_arrays  # noqa: E402
from fullwave.solver.pml_builder import PMLBuilder  # noqa: E402


def _count_cuda_devices() -> int:
    try:
        return cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError:
        return 0


pytestmark = pytest.mark.skipif(_count_cuda_devices() < 1, reason="needs a CUDA device")


@pytest.fixture
def four_devices_with_peer_access_off(monkeypatch):
    monkeypatch.setenv("FULLWAVE_P2P", "off")
    monkeypatch.setattr(cp.cuda.runtime, "getDeviceCount", lambda: 4)


def test_medium_upload_goes_straight_to_device_0_when_peer_access_is_off(
    monkeypatch,
    four_devices_with_peer_access_off,
):
    def fail_if_called(*_args, **_kwargs):
        pytest.fail("multi-GPU upload used while peer access is off")

    monkeypatch.setattr("fullwave.medium._upload_arrays_multi_gpu", fail_if_called)
    rng = np.random.default_rng(0)
    named_arrays = [(f"array_{i}", rng.random((4, 5, 6))) for i in range(5)]

    uploaded = _upload_or_convert_arrays(cp, np.float32, named_arrays)

    for name, array in named_arrays:
        assert uploaded[name].device.id == 0
        np.testing.assert_array_equal(uploaded[name].get(), array.astype(np.float32))


def test_pml_builder_moves_cupy_inputs_to_host_when_peer_access_is_off(
    monkeypatch,
    four_devices_with_peer_access_off,
):
    seen_types = []

    def record_inputs(self, named_arrays, n_gpus):
        seen_types.extend(type(array) for _, array in named_arrays)
        return dict(named_arrays)

    monkeypatch.setattr(PMLBuilder, "_extend_arrays_multi_gpu", record_inputs)
    builder = PMLBuilder.__new__(PMLBuilder)
    builder.medium_org = SimpleNamespace()
    named_arrays = [
        ("sound_speed", cp.ones((4, 5, 6), dtype=cp.float32)),
        ("density", np.ones((4, 5, 6), dtype=np.float32)),
    ]

    builder._extend_arrays_gpu(named_arrays)

    assert seen_types == [np.ndarray, np.ndarray]
