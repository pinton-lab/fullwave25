"""An evaluation marks invalid cells the lookup still serves, and the user is warned."""

import json
import logging
import os
import shutil
from pathlib import Path

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

    def test_every_reason_it_carries_is_one_of_the_six(self):
        held = RelaxationParametersGenerator()
        named = {reason for reasons in held.invalid_reasons().values() for reason in reasons}
        assert named <= {
            "optimization",
            "pml_reflection",
            "attenuation",
            "phase_velocity",
            "diverged",
            "not_fitted",
        }

    def test_a_marked_cell_of_the_shipped_table_reads_as_invalid(self):
        held = RelaxationParametersGenerator()
        cells = sorted(held.invalid_reasons())
        coefficients = np.asarray([one for one, _ in cells], dtype=np.float64)
        exponents = np.asarray([other for _, other in cells], dtype=np.float64)
        assert not np.asarray(held.is_usable(coefficients, exponents)).any()

    def test_another_table_carries_no_record_of_its_own(self, table):
        assert generator_for(table).invalid_cells is None


class TestANotFittedCellMapsToTheNearestFittedExponent:
    """The lookup never uses the zeros a not_fitted cell holds."""

    def test_a_specified_value_on_a_not_fitted_cell_uses_the_nearest_fitted_exponent(self):
        held = RelaxationParametersGenerator()
        used = held.generate(np.array([[5.0]]), np.array([[0.5]]))
        fitted = held.generate(np.array([[5.0]]), np.array([[1.0001]]))
        for key, value in fitted.items():
            np.testing.assert_array_equal(used[key], value, err_msg=key)
        assert np.asarray(used["d_x1_nu1"]).any()

    def test_the_warning_names_the_specified_value_and_the_cell_used(self, caplog):
        with caplog.at_level(logging.WARNING):
            RelaxationParametersGenerator().generate(np.array([[5.0]]), np.array([[0.5]]))
        assert "off the fitted cells of the table" in caplog.text
        assert "(5, 0.5) -> (5, 1.0001)" in caplog.text

    def test_a_value_between_nodes_moves_off_the_not_fitted_cell_it_lands_on(self, caplog):
        held = RelaxationParametersGenerator()
        with caplog.at_level(logging.WARNING):
            used = held.generate(np.array([[1.05]]), np.array([[1.65]]))
        fitted = held.generate(np.array([[1.1]]), np.array([[1.6]]))
        for key, value in fitted.items():
            np.testing.assert_array_equal(used[key], value, err_msg=key)
        assert "(1.05, 1.65) -> (1.1, 1.6)" in caplog.text

    def test_no_cell_of_the_default_grid_uses_the_zeros_of_a_not_fitted_cell(self):
        held = RelaxationParametersGenerator()
        alpha_coeff, alpha_power = np.meshgrid(
            held.alpha_list.ravel(), held.power_list.ravel(), indexing="ij"
        )
        used = held.generate(alpha_coeff, alpha_power)
        strengths = np.stack([np.asarray(used[f"d_x1_nu{index}"]) for index in range(1, 5)])
        assert (strengths != 0).any(axis=0).all()

    def test_on_a_tie_the_lower_exponent_is_used(self, tmp_path):
        power = np.array([[0.5, 1.0, 1.5]], dtype=np.float64)
        database = np.zeros((ALPHA.size, power.size, 4 * MECHANISMS + 2), dtype=np.float64)
        database += np.arange(1, power.size + 1, dtype=np.float64)[None, :, None]
        table = tmp_path / "table.mat"
        savemat(
            table,
            {
                "database": database,
                "alpha_0_list": ALPHA,
                "power_list": power,
                "invalid_matrix": np.zeros((ALPHA.size, power.size), dtype=np.uint8),
            },
        )
        record = write_record(
            tmp_path / "invalid_cells.json", [(1, 1)], power=power, reasons=("not_fitted",)
        )
        used = generator_for(table, record).generate(np.array([[0.2]]), np.array([[1.0]]))
        assert float(np.asarray(used["kappa_x1"]).ravel()[0]) == 1.0

    def test_a_row_with_no_fitted_cell_is_refused(self, tmp_path, table):
        record = write_record(
            tmp_path / "invalid_cells.json", [(0, 0), (0, 1)], reasons=("not_fitted",)
        )
        with pytest.raises(AssertionError, match="no fitted cell"):
            generator_for(table, record)

    def test_a_cycle_of_moves_under_a_band_transfer_is_refused(self, tmp_path):
        """At s = 0.1, row 0.32 sends 1.5 to 1.6, and row 0.26 sends 1.6 back to 1.5."""
        held = RelaxationParametersGenerator()
        alpha = held.alpha_list.ravel().round(10).tolist()
        power = held.power_list.ravel().round(10).tolist()
        marked = [
            (alpha.index(0.32), power.index(1.4)),
            (alpha.index(0.32), power.index(1.5)),
            (alpha.index(0.26), power.index(1.6)),
            (alpha.index(0.26), power.index(1.7)),
        ]
        record = write_record(
            tmp_path / "invalid_cells.json",
            marked,
            alpha=held.alpha_list,
            power=held.power_list,
            reasons=("not_fitted",),
        )
        cycling = RelaxationParametersGenerator(path_invalid_cells=record)
        with pytest.raises(ValueError, match="cycle of moves"):
            cycling.generate(np.array([[1.0]]), np.array([[1.5]]), band_scale=0.1)

    def test_a_second_move_under_a_band_transfer_ends_on_a_fitted_cell(self):
        """(8.0, 1.7) at s = 0.1 moves to 1.6, then its key lands where 1.6 is not fitted."""
        held = RelaxationParametersGenerator()
        used = held.generate(np.array([[8.0]]), np.array([[1.7]]), band_scale=0.1)
        fitted = held.generate(np.array([[8.0]]), np.array([[1.0001]]), band_scale=0.1)
        assert_same_parameters(used, fitted)


class TestAValueTheTableDoesNotHoldWarns:
    """A specified value that is not a node maps to a cell, and the warning names both."""

    def test_a_value_between_nodes_names_the_cell_used(self, caplog):
        with caplog.at_level(logging.WARNING):
            RelaxationParametersGenerator().generate(np.array([[2.05]]), np.array([[1.0]]))
        assert "off the fitted cells of the table" in caplog.text
        assert "(2.05, 1) -> (3, 1.0001)" in caplog.text

    def test_a_value_above_the_axis_names_the_last_node(self, caplog):
        with caplog.at_level(logging.WARNING):
            RelaxationParametersGenerator().generate(np.array([[25.0]]), np.array([[1.0]]))
        assert "(25, 1) -> (20, 1.0001)" in caplog.text

    def test_a_value_on_a_node_does_not_warn(self, caplog):
        with caplog.at_level(logging.WARNING):
            RelaxationParametersGenerator().generate(np.array([[0.5]]), np.array([[1.2]]))
        assert "off the fitted cells" not in caplog.text

    def test_float32_noise_near_a_node_does_not_warn(self, caplog):
        alpha_coeff = np.array([[0.07, 0.5, 1.3]], dtype=np.float32).astype(np.float64)
        alpha_power = np.array([[1.2, 0.7, 1.1]], dtype=np.float32).astype(np.float64)
        assert alpha_coeff[0, 0] != 0.07
        with caplog.at_level(logging.WARNING):
            RelaxationParametersGenerator().generate(alpha_coeff, alpha_power)
        assert "off the fitted cells" not in caplog.text

    def test_two_specified_pairs_on_one_cell_are_both_listed(self, caplog):
        with caplog.at_level(logging.WARNING):
            RelaxationParametersGenerator().generate(
                np.array([[0.101, 0.105]]), np.array([[1.2, 1.2]])
            )
        assert "(0.101, 1.2) -> (0.11, 1.2); (0.105, 1.2) -> (0.11, 1.2)" in caplog.text

    def test_many_cells_are_summarized(self, caplog):
        alpha_coeff = (np.arange(15, dtype=np.float64) * 0.01 + 0.105)[None, :]
        with caplog.at_level(logging.WARNING):
            RelaxationParametersGenerator().generate(alpha_coeff, np.full_like(alpha_coeff, 1.2))
        assert "15 voxels specify" in caplog.text
        assert "and 5 more." in caplog.text

    def test_one_voxel_off_the_grid_is_named_in_the_singular(self, caplog):
        with caplog.at_level(logging.WARNING):
            RelaxationParametersGenerator().generate(np.array([2.05]), np.array([1.0]))
        assert "1 voxel specifies" in caplog.text
        assert "1 voxels" not in caplog.text

    def test_several_voxels_off_the_grid_are_named_in_the_plural(self, caplog):
        with caplog.at_level(logging.WARNING):
            RelaxationParametersGenerator().generate(np.array([2.05, 2.05]), np.array([1.0, 1.0]))
        assert "2 voxels specify" in caplog.text


class TestTheReasonsAreThoseOfTheCellUsed:
    """A value off a node is warned with the reasons of the cell it maps to."""

    def test_a_value_below_an_invalid_node_names_the_reasons_of_that_node(
        self, tmp_path, table, caplog
    ):
        one = generator_for(
            table, write_record(tmp_path / "invalid_cells.json", [(2, 1)], reasons=("diverged",))
        )
        with caplog.at_level(logging.WARNING):
            one.generate(np.array([[0.25]]), np.array([[1.4]]))
        assert "marked 1 voxels invalid" in caplog.text
        assert "(0.3, 1.5, ['diverged'])" in caplog.text


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


def assert_same_parameters(first, second):
    """Assert that two parameter sets are bit-equal, key by key."""
    for key, value in second.items():
        np.testing.assert_array_equal(np.asarray(first[key]), np.asarray(value), err_msg=key)


class TestAZeroCoefficientIsLosslessAndQuiet:
    """A voxel of alpha_0 = 0 is lossless, so no lookup warning names it."""

    @pytest.mark.parametrize(
        ("band_scale", "scale"), [(1.0, False), (1.0, True), (0.1, False), (10.0, True)]
    )
    def test_a_zero_coefficient_logs_no_warning(self, caplog, band_scale, scale):
        with caplog.at_level(logging.WARNING):
            RelaxationParametersGenerator().generate(
                np.array([[0.0, 0.0, 0.0]]),
                np.array([[1.0, 0.5, 1.4]]),
                band_scale=band_scale,
                scale_to_requested_alpha_coeff=scale,
            )
        assert caplog.text == ""

    def test_the_other_voxels_are_unchanged(self):
        held = RelaxationParametersGenerator()
        together = held.generate(np.array([[0.0, 0.5]]), np.array([[1.0, 1.2]]))
        alone = held.generate(np.array([[0.5]]), np.array([[1.2]]))
        assert_same_parameters({key: value[:, 1:] for key, value in together.items()}, alone)

    def test_a_negative_coefficient_still_warns(self, caplog):
        with caplog.at_level(logging.WARNING):
            RelaxationParametersGenerator().generate(np.array([[-1.0]]), np.array([[1.2]]))
        assert "out of range" in caplog.text


class TestTheExponentOneReadsAsTheNode:
    """The axis holds 1.0001 in place of 1.0, and a specified 1.0 maps to it without a warning."""

    def test_an_exponent_of_one_does_not_warn(self, caplog):
        with caplog.at_level(logging.WARNING):
            RelaxationParametersGenerator().generate(np.array([[0.5]]), np.array([[1.0]]))
        assert "off the fitted cells" not in caplog.text

    def test_an_exponent_of_one_uses_the_node(self):
        held = RelaxationParametersGenerator()
        assert_same_parameters(
            held.generate(np.array([[0.5]]), np.array([[1.0]])),
            held.generate(np.array([[0.5]]), np.array([[1.0001]])),
        )

    def test_an_exponent_of_two_does_not_warn(self, caplog):
        with caplog.at_level(logging.WARNING):
            RelaxationParametersGenerator().generate(np.array([[0.5]]), np.array([[2.0]]))
        assert "off the fitted cells" not in caplog.text
        assert "out of range" not in caplog.text

    def test_an_exponent_of_two_uses_the_node(self):
        held = RelaxationParametersGenerator()
        assert_same_parameters(
            held.generate(np.array([[0.5]]), np.array([[2.0]])),
            held.generate(np.array([[0.5]]), np.array([[1.999]])),
        )

    def test_an_exponent_above_two_still_warns(self, caplog):
        with caplog.at_level(logging.WARNING):
            RelaxationParametersGenerator().generate(np.array([[0.5]]), np.array([[2.1]]))
        assert "out of range" in caplog.text
        assert "(0.5, 2.1) -> (0.5, 1.999)" in caplog.text

    def test_an_exponent_of_one_off_the_coefficient_axis_still_warns(self, caplog):
        with caplog.at_level(logging.WARNING):
            RelaxationParametersGenerator().generate(np.array([[0.505]]), np.array([[1.0]]))
        assert "(0.505, 1) -> (0.51, 1.0001)" in caplog.text


class TestTheRecordIsFoundBesideAnyTable:
    """The record named after the table is read wherever the table is."""

    def test_the_default_table_by_a_relative_path_moves_off_a_not_fitted_cell(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        relative = Path(os.path.relpath(ShippedDatabase.table, tmp_path))
        held = RelaxationParametersGenerator(path_database=relative)
        assert held.invalid_cells is not None
        assert_same_parameters(
            held.generate(np.array([[5.0]]), np.array([[0.5]])),
            held.generate(np.array([[5.0]]), np.array([[1.0001]])),
        )

    def test_a_copied_table_reads_the_record_copied_beside_it(self, tmp_path):
        table = tmp_path / ShippedDatabase.table.name
        shutil.copyfile(ShippedDatabase.table, table)
        shutil.copyfile(
            ShippedDatabase.invalid_cells, tmp_path / ShippedDatabase.invalid_cells.name
        )
        held = RelaxationParametersGenerator(path_database=table)
        assert held.invalid_cells is not None
        assert_same_parameters(
            held.generate(np.array([[5.0]]), np.array([[0.5]])),
            held.generate(np.array([[5.0]]), np.array([[1.0001]])),
        )
