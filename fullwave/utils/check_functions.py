"""collections of functions to check the input values."""

from pathlib import Path
from typing import Any


def check_instance(target_var: Any, instances: list[Any] | Any) -> None:  # noqa: ANN401
    """Check whether target_var is an instance of the specified type.

    Raises:
        TypeError: If target_var is not an instance of instance.

    """
    if not isinstance(instances, list):
        instances = [instances]
    if not any(isinstance(target_var, instance) for instance in instances):
        error_message = f"{target_var} is not an instance of {instances}"
        raise TypeError(error_message)


def check_path_exists(target_path: Path) -> None:
    """Check whether the provided target_path exists.

    Raises:
        ValueError: If target_path does not exist.

    """
    if not target_path.exists():
        error_message = f"{target_path} does not exist"
        raise ValueError(error_message)


def check_compatible_value(
    value: Any,  # noqa: ANN401
    compatible_values: list[Any],
    error_message_template: str = "",
) -> None:
    """Check whether the provided value is in the list of compatible values.

    Raises:
        ValueError: If value is not in compatible_values.

    """
    if value not in compatible_values:
        error_message = (
            f"{value} is not compatible. "
            f"Compatible values are: {compatible_values}. "
            f"{error_message_template}"
        )
        raise ValueError(error_message)
