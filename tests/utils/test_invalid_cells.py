"""An evaluation marks invalid cells the lookup still serves, and the user is warned."""

import json
import logging

import numpy as np
import pytest
from scipy.io import savemat

from fullwave.solver.shipped_database import ShippedDatabase
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


def write_record(path, marked, alpha=ALPHA, power=POWER, reasons=("pml_reflection",)):
    """Write an invalid-cell record marking the given row and column pairs."""
    record = {
        "schema": 2,
        "grid": {"alpha_coeff": alpha.ravel().tolist(), "alpha_power": power.ravel().tolist()},
        "invalid": [
            {
                "alpha_coeff": float(alpha.ravel()[row]),
                "alpha_power": float(power.ravel()[column]),
                "row": row,
                "column": column,
                "reasons": list(reasons),
            }
            for row, column in marked
        ],
    }
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return path


MECHANISMS = 2


@pytest.fixture
def table(tmp_path):
    """Return a small lookup table, on its own grid and not the shipped one."""
    return write_table(tmp_path / "table.mat", MECHANISMS)


def generator_for(table, record=None):
    """Return a generator reading that table, and the record where there is one."""
    return RelaxationParametersGenerator(
        n_relaxation_mechanisms=MECHANISMS,
        path_database=table,
        path_invalid_cells=record,
    )


class TestTheGeneratorWithoutARecord:
    """The default behaviour does not change."""

    def test_nothing_is_marked(self, table):
        held = generator_for(table)
        assert held.invalid_cells is None
        assert held.invalid_reasons() == {}
        assert held.is_usable(ALPHA, np.full_like(ALPHA, 1.5)).all()

    def test_no_warning_is_logged(self, table, caplog):
        held = generator_for(table)
        with caplog.at_level(logging.WARNING):
            held.generate(ALPHA, np.full_like(ALPHA, 1.5))
        assert "invalid" not in caplog.text


class TestTheGeneratorWithARecord:
    """A marked cell reads as invalid and the lookup still serves it."""

    def held(self, tmp_path, table, marked, reasons=("pml_reflection",)):
        """Return a generator carrying a record that marks the given cells."""
        return generator_for(
            table,
            write_record(tmp_path / "invalid_cells.json", marked, reasons=reasons),
        )

    def test_a_marked_cell_reads_as_invalid(self, tmp_path, table):
        one = self.held(tmp_path, table, [(2, 1)])
        assert not one.is_usable(np.array([0.3]), np.array([1.5]))[0]

    def test_an_unmarked_cell_reads_as_usable(self, tmp_path, table):
        one = self.held(tmp_path, table, [(2, 1)])
        assert one.is_usable(np.array([0.1]), np.array([1.0]))[0]

    def test_the_record_carries_the_reason_of_each_cell(self, tmp_path, table):
        one = self.held(tmp_path, table, [(2, 1)], reasons=("optimization", "attenuation"))
        assert one.invalid_reasons() == {(0.3, 1.5): ["optimization", "attenuation"]}

    def test_the_lookup_still_serves_a_marked_cell(self, tmp_path, table):
        one = self.held(tmp_path, table, [(2, 1)])
        held = one.generate(np.array([[0.3]]), np.array([[1.5]]))
        assert held
        assert all(np.isfinite(np.asarray(value)).all() for value in held.values())

    def test_the_warning_names_the_cells_the_count_and_the_reasons(self, tmp_path, table, caplog):
        one = self.held(tmp_path, table, [(2, 1)], reasons=("diverged",))
        with caplog.at_level(logging.WARNING):
            one.generate(np.array([[0.3, 0.3, 0.1]]), np.array([[1.5, 1.5, 1.0]]))
        assert "marked 2 voxels invalid" in caplog.text
        assert "over 1 cells" in caplog.text
        assert "(0.3, 1.5, ['diverged'])" in caplog.text

    def test_no_warning_where_no_request_lands_on_a_marked_cell(self, tmp_path, table, caplog):
        one = self.held(tmp_path, table, [(2, 1)])
        with caplog.at_level(logging.WARNING):
            one.generate(np.array([[0.1]]), np.array([[1.0]]))
        assert "invalid" not in caplog.text

    def test_the_record_is_a_different_thing_from_the_calibration_flag(self, tmp_path, table):
        one = self.held(tmp_path, table, [(2, 1)])
        assert not one.is_usable(np.array([0.3]), np.array([1.5]))[0]
        assert one.is_calibrated(np.array([0.3]), np.array([1.5]))[0]


class TestTheShippedTableCarriesItsOwnRecord:
    """The record describes one grid, so it pairs with one table."""

    def test_the_shipped_table_reads_the_shipped_record(self):
        held = RelaxationParametersGenerator()
        assert held.path_database == ShippedDatabase.table
        assert held.invalid_cells is not None
        assert held.invalid_cells["schema"] == 4
        assert held.invalid_reasons()

    def test_every_reason_it_carries_is_one_of_the_five(self):
        held = RelaxationParametersGenerator()
        named = {reason for reasons in held.invalid_reasons().values() for reason in reasons}
        assert named <= {
            "optimization",
            "pml_reflection",
            "attenuation",
            "phase_velocity",
            "diverged",
        }

    def test_a_marked_cell_of_the_shipped_table_reads_as_invalid(self):
        held = RelaxationParametersGenerator()
        cells = sorted(held.invalid_reasons())
        coefficients = np.asarray([one for one, _ in cells], dtype=np.float64)
        exponents = np.asarray([other for _, other in cells], dtype=np.float64)
        assert not np.asarray(held.is_usable(coefficients, exponents)).any()

    def test_another_table_carries_no_record_of_its_own(self, table):
        assert generator_for(table).invalid_cells is None


class TestWhatTheGeneratorRefuses:
    """A record that describes another grid cannot be applied to this table."""

    def test_a_record_that_is_not_there_is_refused(self, tmp_path, table):
        with pytest.raises(FileNotFoundError):
            generator_for(table, tmp_path / "absent.json")

    def test_a_record_on_another_coefficient_axis_is_refused(self, tmp_path, table):
        path = write_record(
            tmp_path / "other.json", [], alpha=np.array([[9.0, 8.0, 7.0]]), power=POWER
        )
        with pytest.raises(ValueError, match="different alpha_coeff axis"):
            generator_for(table, path)

    def test_a_record_on_another_exponent_axis_is_refused(self, tmp_path, table):
        path = write_record(tmp_path / "other.json", [], alpha=ALPHA, power=np.array([[0.4, 0.6]]))
        with pytest.raises(ValueError, match="different alpha_power axis"):
            generator_for(table, path)
