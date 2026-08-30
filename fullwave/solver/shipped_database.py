"""The relaxation tables this package ships, named in one place.

Every default in the package names these values, so a new calibration is adopted
by editing ``stems`` and ``default_mechanisms`` here and nowhere else. Before this
file the same path was written out in nine modules.

This module imports the standard library alone, so any module may import it.
"""

from pathlib import Path
from typing import ClassVar


class ShippedDatabase:
    """Where each shipped relaxation table lives, and what it holds.

    Attributes
    ----------
    root : Path
        The directory the tables and their records sit in.
    default_mechanisms : int
        The mechanism count a caller gets without naming one.
    stems : dict[int, str]
        The name each count's table and record share.
    mechanisms : int
        The default count, kept as its own name because most callers read it.
    stem : str
        The name the default count's two files share.
    table : Path
        The default count's lookup table.
    invalid_cells : Path
        The record of the cells an evaluation found invalid, and why each one is
        invalid. It is a sibling file because the table is pinned by hash.

    """

    root = Path(__file__).parent / "bins" / "database"
    default_mechanisms = 4
    stems: ClassVar[dict[int, str]] = {
        3: "relaxation_params_database_num_relax=3_20260828_1513",
        4: "relaxation_params_database_num_relax=4_20260828_1456",
    }

    @classmethod
    def stem_of(cls, mechanisms: int) -> str:
        """Return the name one count's table and record share.

        Parameters
        ----------
        mechanisms : int
            How many relaxation mechanisms the table was fitted with.

        Returns
        -------
        str
            The shared name.

        Raises
        ------
        KeyError
            The release ships no table for that count.

        """
        if mechanisms not in cls.stems:
            message = (
                f"No shipped table for {mechanisms} relaxation mechanisms. "
                f"This release ships {sorted(cls.stems)}."
            )
            raise KeyError(message)
        return cls.stems[mechanisms]

    @classmethod
    def table_of(cls, mechanisms: int) -> Path:
        """Return one count's lookup table.

        Parameters
        ----------
        mechanisms : int
            How many relaxation mechanisms the table was fitted with.

        Returns
        -------
        Path
            The table.

        """
        return cls.root / f"{cls.stem_of(mechanisms)}.mat"

    @classmethod
    def invalid_cells_of(cls, mechanisms: int) -> Path:
        """Return one count's record of the cells an evaluation refused.

        Parameters
        ----------
        mechanisms : int
            How many relaxation mechanisms the table was fitted with.

        Returns
        -------
        Path
            The record.

        """
        return cls.root / f"{cls.stem_of(mechanisms)}_invalid_cells.json"

    mechanisms = default_mechanisms
    stem = stems[default_mechanisms]
    table = root / f"{stem}.mat"
    invalid_cells = root / f"{stem}_invalid_cells.json"
