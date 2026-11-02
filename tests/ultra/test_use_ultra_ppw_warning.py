import warnings

import pytest

from fullwave.ultra import backend


def test_ppw_above_speed_optimum_warns_and_still_runs(build_solver):
    with pytest.warns(UserWarning, match=r"optimized for speed at ppw=6; ppw=8 runs"):
        solver = build_solver(ppw=8)

    assert solver.ultra is not None
    assert solver.run(generate_input_only=True) is not None


def test_ppw_at_speed_optimum_does_not_warn(build_solver):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warnings.simplefilter("ignore", backend.FullwaveUltraExperimentalWarning)
        build_solver(ppw=6)


def test_ppw_above_speed_optimum_warns_once(build_solver):
    with pytest.warns(UserWarning, match=r"optimized for speed at ppw=6; ppw=8 runs"):
        build_solver(ppw=8)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warnings.simplefilter("ignore", backend.FullwaveUltraExperimentalWarning)
        build_solver(ppw=8)
