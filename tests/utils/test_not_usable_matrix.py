"""An evaluation marks cells the lookup still serves, and the user is warned."""

import logging

import numpy as np
import pytest
from scipy.io import savemat

from fullwave.utils.relaxation_parameters import RelaxationParametersGenerator

ALPHA = np.array([[0.1, 0.2, 0.3]], dtype=np.float64)
POWER = np.array([[1.0, 1.5]], dtype=np.float64)


def write_table(path, mechanisms=2):
    """Write a small lookup table on the ALPHA by POWER grid."""
    columns = 4 * mechanisms + 2
    savemat(
        path,
        {
            "database": np.ones((ALPHA.size, POWER.size, columns), dtype=np.float64),
            "alpha_0_list": ALPHA,
            "power_list": POWER,
            "invalid_matrix": np.zeros((ALPHA.size, POWER.size), dtype=np.uint8),
        },
    )
    return path


def write_mask(path, marked, alpha=ALPHA, power=POWER):
    """Write a not-usable mask, marking the given row and column pairs."""
    held = np.zeros((alpha.size, power.size), dtype=np.uint8)
    for row, column in marked:
        held[row, column] = 1
    savemat(path, {"alpha_0_list": alpha, "power_list": power, "not_usable_matrix": held})
    return path


@pytest.fixture
def table(tmp_path):
    """Return a small lookup table."""
    return write_table(tmp_path / "table.mat")


class TestTheGeneratorWithoutAMask:
    """The default behaviour does not change."""

    def test_nothing_is_marked(self, table):
        held = RelaxationParametersGenerator(path_database=table)
        assert held.not_usable_matrix is None
        assert held.is_usable(ALPHA, np.full_like(ALPHA, 1.5)).all()

    def test_no_warning_is_logged(self, table, caplog):
        held = RelaxationParametersGenerator(path_database=table)
        with caplog.at_level(logging.WARNING):
            held.generate(ALPHA, np.full_like(ALPHA, 1.5))
        assert "not usable" not in caplog.text


class TestTheGeneratorWithAMask:
    """A marked cell reads as not usable and the lookup still serves it."""

    def held(self, tmp_path, table, marked):
        """Return a generator carrying a mask that marks the given cells."""
        return RelaxationParametersGenerator(
            path_database=table,
            path_not_usable_matrix=write_mask(tmp_path / "mask.mat", marked),
        )

    def test_a_marked_cell_reads_as_not_usable(self, tmp_path, table):
        one = self.held(tmp_path, table, [(2, 1)])
        assert not one.is_usable(np.array([0.3]), np.array([1.5]))[0]

    def test_an_unmarked_cell_reads_as_usable(self, tmp_path, table):
        one = self.held(tmp_path, table, [(2, 1)])
        assert one.is_usable(np.array([0.1]), np.array([1.0]))[0]

    def test_the_lookup_still_serves_a_marked_cell(self, tmp_path, table):
        one = self.held(tmp_path, table, [(2, 1)])
        held = one.generate(np.array([[0.3]]), np.array([[1.5]]))
        assert held
        assert all(np.isfinite(np.asarray(value)).all() for value in held.values())

    def test_the_warning_names_the_cells_and_the_count(self, tmp_path, table, caplog):
        one = self.held(tmp_path, table, [(2, 1)])
        with caplog.at_level(logging.WARNING):
            one.generate(np.array([[0.3, 0.3, 0.1]]), np.array([[1.5, 1.5, 1.0]]))
        assert "marked 2 voxels as not usable" in caplog.text
        assert "over 1 cells" in caplog.text
        assert "(0.3, 1.5)" in caplog.text

    def test_no_warning_where_no_request_lands_on_a_marked_cell(self, tmp_path, table, caplog):
        one = self.held(tmp_path, table, [(2, 1)])
        with caplog.at_level(logging.WARNING):
            one.generate(np.array([[0.1]]), np.array([[1.0]]))
        assert "not usable" not in caplog.text

    def test_the_mask_is_a_different_thing_from_the_calibration_flag(self, tmp_path, table):
        one = self.held(tmp_path, table, [(2, 1)])
        assert not one.is_usable(np.array([0.3]), np.array([1.5]))[0]
        assert one.is_calibrated(np.array([0.3]), np.array([1.5]))[0]


class TestWhatTheGeneratorRefuses:
    """A mask that describes another grid cannot be applied to this table."""

    def test_a_mask_that_is_not_there_is_refused(self, tmp_path, table):
        with pytest.raises(FileNotFoundError):
            RelaxationParametersGenerator(
                path_database=table, path_not_usable_matrix=tmp_path / "absent.mat"
            )

    def test_a_mask_on_another_coefficient_axis_is_refused(self, tmp_path, table):
        path = write_mask(
            tmp_path / "other.mat", [], alpha=np.array([[9.0, 8.0, 7.0]]), power=POWER
        )
        with pytest.raises(ValueError, match="different alpha_0_list"):
            RelaxationParametersGenerator(path_database=table, path_not_usable_matrix=path)

    def test_a_mask_on_another_exponent_axis_is_refused(self, tmp_path, table):
        path = write_mask(tmp_path / "other.mat", [], alpha=ALPHA, power=np.array([[0.4, 0.6]]))
        with pytest.raises(ValueError, match="different power_list"):
            RelaxationParametersGenerator(path_database=table, path_not_usable_matrix=path)
