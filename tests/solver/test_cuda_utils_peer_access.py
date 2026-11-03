import pytest

from fullwave.solver.cuda_utils import (
    binary_reads_peer_access_switch,
    peer_access_is_allowed,
    read_peer_access_mode,
)


@pytest.mark.parametrize(
    ("environ", "expected"),
    [({}, "auto"), ({"FULLWAVE_P2P": ""}, "auto"), ({"FULLWAVE_P2P": " OFF "}, "off")],
)
def test_read_peer_access_mode_accepts_unset_empty_and_padded_values(environ, expected):
    assert read_peer_access_mode(environ=environ) == expected


def test_read_peer_access_mode_rejects_an_unknown_value():
    with pytest.raises(ValueError, match="FULLWAVE_P2P"):
        read_peer_access_mode(environ={"FULLWAVE_P2P": "maybe"})


@pytest.mark.parametrize(
    ("mode", "numa_nodes", "expected"),
    [
        ("off", {0: 0, 1: 0}, False),
        ("on", {0: 0, 1: 1}, True),
        ("auto", {0: 0, 1: 0}, True),
        ("auto", {0: 0, 1: 1}, False),
        ("auto", {}, True),
        ("auto", {0: 0, 2: 1}, False),
    ],
)
def test_peer_access_is_allowed_follows_mode_and_numa_nodes(mode, numa_nodes, expected):
    assert peer_access_is_allowed(mode=mode, numa_nodes=numa_nodes) is expected


def test_binary_reads_peer_access_switch_looks_for_the_variable_name(tmp_path):
    binary_reads_peer_access_switch.cache_clear()
    with_switch = tmp_path / "with_switch"
    with_switch.write_bytes(b"\x00ICMAT_MEMORY_SAVING\x00FULLWAVE_P2P\x00")
    without_switch = tmp_path / "without_switch"
    without_switch.write_bytes(b"\x00ICMAT_MEMORY_SAVING\x00")
    assert binary_reads_peer_access_switch(path=str(with_switch))
    assert not binary_reads_peer_access_switch(path=str(without_switch))
