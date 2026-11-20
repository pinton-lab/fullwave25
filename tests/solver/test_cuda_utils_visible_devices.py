import pytest

from fullwave.solver.cuda_utils import cuda_visible_devices_for_child


def test_ids_are_unchanged_without_visible_devices():
    assert cuda_visible_devices_for_child(device_ids=[2, 3], environ={}) == "2,3"


def test_ids_count_within_the_visible_devices():
    environ = {"CUDA_VISIBLE_DEVICES": "2,3"}
    assert cuda_visible_devices_for_child(device_ids=[0, 1], environ=environ) == "2,3"


def test_ids_can_reorder_the_visible_devices():
    environ = {"CUDA_VISIBLE_DEVICES": "4, 6, 7"}
    assert cuda_visible_devices_for_child(device_ids=[2, 0], environ=environ) == "7,4"


def test_uuid_entries_pass_through():
    environ = {"CUDA_VISIBLE_DEVICES": "GPU-aaaa,GPU-bbbb"}
    assert cuda_visible_devices_for_child(device_ids=[1], environ=environ) == "GPU-bbbb"


def test_an_id_beyond_the_visible_devices_names_both_lists():
    environ = {"CUDA_VISIBLE_DEVICES": "2,3"}
    with pytest.raises(ValueError, match=r"\[2\] counts within CUDA_VISIBLE_DEVICES=2,3"):
        cuda_visible_devices_for_child(device_ids=[2], environ=environ)


def test_an_empty_visible_devices_variable_gives_the_child_no_gpu():
    environ = {"CUDA_VISIBLE_DEVICES": ""}
    assert cuda_visible_devices_for_child(device_ids=[0], environ=environ) == ""
