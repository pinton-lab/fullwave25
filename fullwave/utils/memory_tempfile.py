"""memory-tempfile.

adapted from
https://github.com/mbello/memory-tempfile/commit/c06b5405672435d861ac1d47db77c8067a6de77a
"""

import logging
import os
import platform
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path

MEM_BASED_FS = ["tmpfs", "ramfs"]
SUITABLE_PATHS = ["/tmp", "/run/user/{uid}", "/run/shm", "/dev/shm"]  # noqa: S108

logger = logging.getLogger("__main__." + __name__)


class MemoryTempfile:
    """MemoryTempfile class for creating temporary files in memory-based filesystems.

    This class is a wrapper around the tempfile module to provide
    functionality for creating temporary files in memory-based filesystems
    (e.g., tmpfs, ramfs) on Linux systems.
    It allows users to specify preferred paths, remove certain paths,
    and add additional paths for temporary files.
    The class also provides methods to check if a memory temporary directory
    is found, get the usable memory temporary directory paths,
    and create temporary files and directories.
    """

    def __init__(  # noqa: C901, PLR0912
        self,
        *,
        preferred_paths: list | None = None,
        remove_paths: list | bool | None = None,
        additional_paths: list | None = None,
        filesystem_types: list | None = None,
        fallback: str | bool | None = None,
    ) -> None:
        """Initialize MemoryTempfile instance.

        Parameters
        ----------
        preferred_paths : list, optional
            List of preferred paths for temporary files.
        remove_paths : list | bool, optional
            List of paths to remove from the suitable paths.
            If True, all suitable paths are removed.
        additional_paths : list, optional
            List of additional paths to include in the suitable paths.
        filesystem_types : list, optional
            List of filesystem types to consider as memory-based.
        fallback : str | bool, optional
            Fallback path to use if no memory temporary directory is found.
            If True, the fallback is set to the system's temporary directory.
            If False, no fallback is used.

        Raises
        ------
        RuntimeError
            If no memory temporary directory is found and fallback is disabled.

        """
        self.os_tempdir = tempfile.gettempdir()
        suitable_paths = [self.os_tempdir, *SUITABLE_PATHS]

        if isinstance(fallback, bool):
            self.fallback = self.os_tempdir if fallback else None
        else:
            self.fallback = fallback

        if platform.system() == "Linux":
            self.filesystem_types = (
                list(filesystem_types) if filesystem_types is not None else MEM_BASED_FS
            )

            preferred_paths = [] if preferred_paths is None else preferred_paths

            if isinstance(remove_paths, bool) and remove_paths:
                suitable_paths = []
            elif isinstance(remove_paths, list) and len(remove_paths) > 0:
                suitable_paths = [i for i in suitable_paths if i not in remove_paths]

            additional_paths = [] if additional_paths is None else additional_paths

            self.suitable_paths = preferred_paths + suitable_paths + additional_paths

            uid = os.geteuid()

            with Path("/proc/self/mountinfo").open(encoding="utf-8") as file:
                mnt_info = {i[2]: i for i in [line.split() for line in file]}

            self.usable_paths: OrderedDict = OrderedDict()
            for path in self.suitable_paths:
                path_uid = path.replace("{uid}", str(uid))

                # We may have repeated
                if self.usable_paths.get(path_uid) is not None:
                    continue
                self.usable_paths[path_uid] = False
                try:
                    dev = Path(path_uid).stat().st_dev
                    major, minor = os.major(dev), os.minor(dev)
                    mp = mnt_info.get(f"{major}:{minor}")
                    if mp and mp[mp.index("-", 6) + 1] in self.filesystem_types:
                        self.usable_paths[path_uid] = mp
                except FileNotFoundError:
                    pass

            for key in [k for k, v in self.usable_paths.items() if not v]:
                del self.usable_paths[key]

            if len(self.usable_paths) > 0:
                self.tempdir = Path(next(iter(self.usable_paths.keys())))
            elif self.fallback is not None:
                self.tempdir = Path(self.fallback)
            else:
                error_message = "No memory temporary dir found and fallback is disabled.\n"
                logger.error(error_message)
                raise RuntimeError(error_message)

    def found_mem_tempdir(self) -> bool:
        """Check if any memory temporary directory is found.

        Returns
        -------
        bool
            True if any memory temporary directory is found, False otherwise.

        """
        return len(self.usable_paths) > 0

    def using_mem_tempdir(self) -> bool:
        """Check if the current temporary directory is a memory temporary directory.

        Returns
        -------
        bool
            True if the current temporary directory is a memory temporary directory,
            False otherwise.

        """
        return self.tempdir in self.usable_paths

    def get_usable_mem_tempdir_paths(self) -> list:
        """Get the list of usable memory temporary directory paths.

        Returns
        -------
        list
            List of usable memory temporary directory paths.

        """
        return list(self.usable_paths.keys())

    def gettempdir(self) -> Path:
        """Get the current temporary directory.

        Returns
        -------
        Path
            The current temporary directory.

        """
        return self.tempdir

    def gettempdirb(self) -> bytes:
        """Get the current temporary directory as bytes.

        Returns
        -------
        bytes
            The current temporary directory as bytes.

        """
        return str(self.tempdir).encode(sys.getfilesystemencoding(), "surrogateescape")

    def mkdtemp(
        self,
        suffix: str | None = None,
        prefix: str | None = None,
        dir: str | None = None,  # noqa: A002
    ) -> str:
        """Create a temporary directory.

        Parameters
        ----------
        suffix : str, optional
            Suffix for the temporary directory name.
        prefix : str, optional
            Prefix for the temporary directory name.
        dir : str, optional
            Directory where the temporary directory will be created.

        Returns
        -------
        str
            The name of the created temporary directory.

        """
        return tempfile.mkdtemp(suffix=suffix, prefix=prefix, dir=dir or self.tempdir)

    def mkstemp(
        self,
        suffix: str | None = None,
        prefix: str | None = None,
        dir: str | None = None,  # noqa: A002
        *,
        text: bool = False,
    ) -> tuple[int, str]:
        """Create a temporary file using mkstemp and return a tuple (file descriptor, file name).

        Parameters
        ----------
        suffix : str, optional
            Suffix for the temporary file name.
        prefix : str, optional
            Prefix for the temporary file name.
        dir : str, optional
            Directory where the temporary file will be created.
        text : bool, optional
            If True, the file will be opened in text mode (default is False).

        Returns
        -------
        tuple[int, str]
            A tuple containing the file descriptor and the name of the created temporary file.

        """
        return tempfile.mkstemp(
            suffix=suffix,
            prefix=prefix,
            dir=dir or self.tempdir,
            text=text,
        )

    def TemporaryDirectory(  # noqa: N802
        self,
        suffix: str | None = None,
        prefix: str | None = None,
        dir: str | None = None,  # noqa: A002
    ) -> tempfile.TemporaryDirectory[str]:
        """Create a temporary directory and return a TemporaryDirectory object.

        Parameters
        ----------
        suffix : str, optional
            Suffix for the temporary file name.
        prefix : str, optional
            Prefix for the temporary file name.
        dir : str, optional
            Directory where the temporary file will be created.

        Returns
        -------
        tempfile.TemporaryDirectory[str]
            A TemporaryDirectory object representing the created temporary directory.

        """
        return tempfile.TemporaryDirectory(
            suffix=suffix,
            prefix=prefix,
            dir=dir or self.tempdir,
        )

    def SpooledTemporaryFile(  # noqa: N802
        self,
        max_size: int = 0,
        mode: str = "w+b",
        buffering: int = -1,
        encoding: str | None = None,
        newline: str | None = None,
        suffix: str | None = None,
        prefix: str | None = None,
        dir: str | None = None,  # noqa: A002
    ) -> tempfile.SpooledTemporaryFile:
        """Create a spooled temporary file.

        Parameters
        ----------
        max_size : int, optional
            Maximum size of the file before it is moved to disk.
        mode : str, optional
            Mode in which the file is opened (default is 'w+b').
        buffering : int, optional
            Buffering policy (default is -1, which means the default buffering).
        encoding : str, optional
            Encoding of the file (default is None).
        newline : str, optional
            Newline handling (default is None).
        suffix : str, optional
            Suffix for the temporary file name (default is None).
        prefix : str, optional
            Prefix for the temporary file name (default is None).
        dir : str, optional
            Directory where the temporary file will be created (default is None).

        Returns
        -------
        tempfile.SpooledTemporaryFile
            A SpooledTemporaryFile object representing the created temporary file.

        Notes
        -----
        Temporary file wrapper, specialized to switch from BytesIO
        or StringIO to a real file when it exceeds a certain size or
        when a fileno is needed.

        """
        return tempfile.SpooledTemporaryFile(
            max_size=max_size,
            mode=mode,
            buffering=buffering,
            encoding=encoding,
            newline=newline,
            suffix=suffix,
            prefix=prefix,
            dir=dir or self.tempdir,
        )

    def NamedTemporaryFile(  # noqa: N802
        self,
        mode: str = "w+b",
        buffering: int = -1,
        encoding: str | None = None,
        newline: str | None = None,
        suffix: str | None = None,
        prefix: str | None = None,
        dir: str | None = None,  # noqa: A002
        *,
        delete: bool = True,
    ) -> tempfile.NamedTemporaryFile:
        """Create and return a temporary file.

        Parameters
        ----------
        mode : str, optional
            Mode in which the file is opened (default is 'w+b').
        buffering : int, optional
            Buffering policy (default is -1, which means the default buffering).
        encoding : str, optional
            Encoding of the file (default is None).
        newline : str, optional
            Newline handling (default is None).
        suffix : str, optional
            Suffix for the temporary file name (default is None).
        prefix : str, optional
            Prefix for the temporary file name (default is None).
        dir : str, optional
            Directory where the temporary file will be created (default is None).
        delete : bool, optional
            If True, the file will be deleted when closed (default is True).

        m        Returns
        -------
        tempfile.NamedTemporaryFile
            A NamedTemporaryFile object representing the created temporary file.

        """
        return tempfile.NamedTemporaryFile(
            mode=mode,
            buffering=buffering,
            encoding=encoding,
            newline=newline,
            suffix=suffix,
            prefix=prefix,
            dir=dir or self.tempdir,
            delete=delete,
        )

    def TemporaryFile(  # noqa: N802
        self,
        mode: str = "w+b",
        buffering: int = -1,
        encoding: str | None = None,
        newline: str | None = None,
        suffix: str | None = None,
        prefix: str | None = None,
        dir: str | None = None,  # noqa: A002
    ) -> tempfile.TemporaryFile:
        """Create and return a temporary file.

        Parameters
        ----------
        mode : str, optional
            Mode in which the file is opened (default is 'w+b').
        buffering : int, optional
            Buffering policy (default is -1, which means the default buffering).
        encoding : str, optional
            Encoding of the file (default is None).
        newline : str, optional
            Newline handling (default is None).
        suffix : str, optional
            Suffix for the temporary file name (default is None).
        prefix : str, optional
            Prefix for the temporary file name (default is None).
        dir : str, optional
            Directory where the temporary file will be created (default is None).

        Returns
        -------
        tempfile.TemporaryFile
            A TemporaryFile object representing the created temporary file.

        -----

        """
        return tempfile.TemporaryFile(
            mode=mode,
            buffering=buffering,
            encoding=encoding,
            newline=newline,
            suffix=suffix,
            prefix=prefix,
            dir=dir or self.tempdir,
        )

    @staticmethod
    def gettempprefix() -> str:
        """Return the default temporary directory.

        Returns
        -------
        str
            The default temporary directory.

        """
        return tempfile.gettempdir()

    @staticmethod
    def gettempprefixb() -> bytes:
        """Return the default temporary directory as bytes.

        Returns
        -------
        bytes
            The default temporary directory as bytes.

        """
        return tempfile.gettempprefixb()
