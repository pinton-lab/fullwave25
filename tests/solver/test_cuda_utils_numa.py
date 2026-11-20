import pytest

from fullwave.solver import cuda_utils
from fullwave.solver.cuda_utils import (
    first_consecutive_gpus_on_different_nodes,
    numa_nodes_of_cuda_ordinals,
    read_numa_node,
)


def _write_node(root, address, text):
    folder = root / address
    folder.mkdir()
    (folder / "numa_node").write_text(text)


def _fake_driver(monkeypatch, *, pci_bus_id_of_ordinal):
    def fake_device_get(device, ordinal):
        device._obj.value = ordinal

    def fake_pci_bus_id(buffer, _length, device):
        buffer.value = pci_bus_id_of_ordinal[device.value].encode()

    monkeypatch.setattr(cuda_utils, "cuInit", lambda _flags: 0)
    monkeypatch.setattr(cuda_utils, "cuDeviceGet", fake_device_get)
    monkeypatch.setattr(cuda_utils, "cuDeviceGetPCIBusId", fake_pci_bus_id)


def test_read_numa_node_maps_driver_address_to_sysfs_address(tmp_path):
    _write_node(tmp_path, "0000:3b:00.0", "1\n")
    assert read_numa_node("00000000:3B:00.0", sysfs_root=tmp_path) == 1


def test_read_numa_node_returns_minus_one_when_unknown(tmp_path):
    _write_node(tmp_path, "0000:af:00.0", "-1\n")
    assert read_numa_node("00000000:AF:00.0", sysfs_root=tmp_path) == -1


def test_read_numa_node_raises_when_device_is_missing(tmp_path):
    with pytest.raises(OSError, match="numa_node"):
        read_numa_node("00000000:3B:00.0", sysfs_root=tmp_path)


def test_numa_nodes_of_cuda_ordinals_follows_the_driver_order_not_the_pci_order(
    tmp_path,
    monkeypatch,
):
    _write_node(tmp_path, "0000:02:00.0", "0\n")
    _write_node(tmp_path, "0000:82:00.0", "1\n")
    _fake_driver(
        monkeypatch,
        pci_bus_id_of_ordinal={0: "00000000:82:00.0", 1: "00000000:02:00.0"},
    )
    nodes = numa_nodes_of_cuda_ordinals(ordinals=[0, 1], sysfs_root=tmp_path)
    assert nodes == {0: 1, 1: 0}


def test_numa_nodes_of_cuda_ordinals_leaves_out_unknown_nodes(tmp_path, monkeypatch):
    _write_node(tmp_path, "0000:02:00.0", "0\n")
    _write_node(tmp_path, "0000:82:00.0", "-1\n")
    _fake_driver(
        monkeypatch,
        pci_bus_id_of_ordinal={0: "00000000:02:00.0", 1: "00000000:82:00.0"},
    )
    assert numa_nodes_of_cuda_ordinals(ordinals=[0, 1], sysfs_root=tmp_path) == {0: 0}


def test_numa_nodes_of_cuda_ordinals_is_empty_when_the_driver_fails(tmp_path, monkeypatch):
    def failing_init(_flags):
        raise RuntimeError("cuInit failed with error code 100")

    monkeypatch.setattr(cuda_utils, "cuInit", failing_init)
    assert numa_nodes_of_cuda_ordinals(ordinals=[0, 1], sysfs_root=tmp_path) == {}


@pytest.mark.parametrize(
    ("device_ids", "numa_nodes", "expected"),
    [
        ([0, 1, 2, 3], {0: 0, 1: 0, 2: 1, 3: 1}, (1, 2)),
        ([0, 1], {0: 1, 1: 1}, None),
        ([0, 1, 2], {0: 0, 2: 1}, (0, 2)),
        ([0, 1], {}, None),
    ],
)
def test_first_consecutive_gpus_on_different_nodes_skips_unknown_nodes(
    device_ids,
    numa_nodes,
    expected,
):
    pair = first_consecutive_gpus_on_different_nodes(device_ids=device_ids, numa_nodes=numa_nodes)
    assert pair == expected
