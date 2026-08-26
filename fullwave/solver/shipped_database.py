"""The relaxation table this package ships, named in one place.

Every default in the package names these values, so a new calibration is adopted
by editing ``stem`` and ``mechanisms`` here and nowhere else. Before this file
the same path was written out in nine modules.

This module imports the standard library alone, so any module may import it.
"""

from pathlib import Path


class ShippedDatabase:
    """Where the shipped relaxation table lives, and what it holds.

    Attributes
    ----------
    root : Path
        The directory the table and its record sit in.
    stem : str
        The name both files share.
    mechanisms : int
        How many relaxation mechanisms the table was fitted with.
    table : Path
        The lookup table itself.
    invalid_cells : Path
        The record of the cells an evaluation found invalid, and why each one is
        invalid. It is a sibling file because the table is pinned by hash.

    """

    root = Path(__file__).parent / "bins" / "database"
    stem = "relaxation_params_database_num_relax=4_20260825_1203"
    mechanisms = 4
    table = root / f"{stem}.mat"
    invalid_cells = root / f"{stem}_invalid_cells.json"
