"""Native-only options of Solver(use_ultra=True) warn once and are otherwise ignored."""

import inspect
import warnings

import pytest

import fullwave
from fullwave.solver.launcher import Launcher
from fullwave.ultra import backend

NON_DEFAULT_OPTIONS = [
    ("pml_layer_thickness_px", 10),
    ("n_transition_layer", 3),
    ("pml_alpha_entrance", 0.5),
    ("exponential_attenuation_pml_thickness_px", 12),
    ("use_gpu_pml", True),
    ("save_gpu_memory", True),
    ("write_maps_the_solver_does_not_read", True),
    ("use_isotropic_relaxation", False),
]
SPONGE_OPTIONS = ["pml_layer_thickness_px", "n_transition_layer", "pml_alpha_entrance"]


@pytest.fixture(autouse=True)
def _no_gpu_query_for_native_solver(monkeypatch):
    monkeypatch.setattr(Launcher, "_verify_cuda_devices_exist", lambda _: None)


def ultra_messages(caught):
    return [str(w.message) for w in caught if "use_ultra" in str(w.message)]


def test_native_only_defaults_match_the_solver_signature():
    parameters = inspect.signature(fullwave.Solver).parameters
    defaults = {name: option.solver_default for name, option in backend.NATIVE_ONLY_OPTIONS.items()}
    assert defaults == {name: parameters[name].default for name in defaults}


def test_no_native_only_option_gives_no_ignored_warning(build_solver):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build_solver()
    assert not [m for m in ultra_messages(caught) if "ignored" in m or "frequency" in m]


@pytest.mark.parametrize(("option", "value"), NON_DEFAULT_OPTIONS)
def test_native_only_option_warns_that_it_is_ignored(build_solver, option, value):
    with pytest.warns(UserWarning, match=rf"{option} is ignored with use_ultra=True"):
        build_solver(**{option: value})


@pytest.mark.parametrize("option", SPONGE_OPTIONS)
def test_pml_option_warning_names_the_sponge_argument(build_solver, option):
    with pytest.warns(UserWarning, match="ultra_sponge_thickness_px"):
        build_solver(**{option: dict(NON_DEFAULT_OPTIONS)[option]})


def test_old_sponge_argument_warns_and_names_the_new_one(build_solver):
    with pytest.warns(UserWarning, match="exponential_attenuation_pml_thickness_px.*ultra_sponge"):
        solver = build_solver(exponential_attenuation_pml_thickness_px=12)
    assert solver.ultra.nbdy == 24


def test_listed_devices_run_on_the_first_and_the_warning_names_it(build_solver):
    with pytest.warns(UserWarning, match="runs on GPU 1 only"):
        solver = build_solver(cuda_device_id=[1, 0])
    assert solver.ultra.device == 1


def test_relaxation_model_request_warns_with_the_difference(build_solver):
    with pytest.warns(UserWarning, match="frequency-independent.*frequency-dependent"):
        build_solver(use_exponential_attenuation=False)


def test_exponential_attenuation_true_is_silent(build_solver):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build_solver(use_exponential_attenuation=True)
    assert not [m for m in ultra_messages(caught) if "frequency" in m]


def test_warning_comes_once_per_process(build_solver):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build_solver(n_transition_layer=3)
        build_solver(n_transition_layer=3)
    assert sum("n_transition_layer is ignored" in m for m in ultra_messages(caught)) == 1


def test_sponge_argument_without_use_ultra_warns_and_names_native_options(build_solver):
    with pytest.warns(UserWarning, match="ultra_sponge_thickness_px is ignored without use_ultra"):
        build_solver(use_ultra=False)


@pytest.mark.parametrize(
    "solver_options",
    [
        {"use_ultra": False},
        {"n_transition_layer": 3},
        {"cuda_device_id": [1, 0]},
        {"use_exponential_attenuation": False},
        {"ppw": 8},
    ],
)
def test_every_warning_points_at_the_line_of_the_caller(solver_arguments, solver_options):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fullwave.Solver(**solver_arguments(**solver_options))
    warned = [w for w in caught if "use_ultra" in str(w.message)]
    assert warned
    assert all(w.filename == __file__ for w in warned)
