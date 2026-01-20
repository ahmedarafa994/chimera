import datetime
import os

import colorama

# Initialize colorama for cross-platform colored terminal text
colorama.init(autoreset=True)

# Verbosity levels (will be imported from config.py in practice)
VERBOSE_NONE = 0  # Only show critical information
VERBOSE_NORMAL = 1  # Show important processes
VERBOSE_DETAILED = 2  # Show all details

# Global verbosity setting (imported from config.py in practice)
VERBOSE_LEVEL = VERBOSE_NORMAL

# Map verbosity levels to names for display
VERBOSE_LEVEL_NAMES = {
    VERBOSE_NONE: "Minimal",
    VERBOSE_NORMAL: "Normal",
    VERBOSE_DETAILED: "Detailed",
}


def log(message, type="info", verbose_level=VERBOSE_NORMAL) -> None:
    """Enhanced logging function with color coding and verbosity control.

    Args:
        message (str): The message to log
        type (str): The type of message (info, success, error, warning, debug, config, result)
        verbose_level (int): The verbosity level of this message

    """
    # Import here to avoid circular imports
    from .config import VERBOSE_LEVEL

    # Skip if verbosity level is too low
    if verbose_level > VERBOSE_LEVEL:
        return

    timestamp = datetime.datetime.now().strftime("%H:%M:%S")

    # Color coding based on message type
    if type in {"info", "success", "error", "warning", "debug", "config", "result"}:
        pass
    else:
        f"[{type.upper()}]"

    # Print to console

    # Trigger callbacks if any
    for callback in _log_callbacks:
        try:
            callback({"timestamp": timestamp, "type": type, "message": message})
        except Exception:
            pass  # Fail silently to avoid breaking the log flow


# Callback management
_log_callbacks = []


def register_log_callback(callback) -> None:
    """Register a callback function to receive log events."""
    _log_callbacks.append(callback)


def clear_log_callbacks() -> None:
    """Clear all registered callbacks."""
    global _log_callbacks
    _log_callbacks = []


def print_header(title, width=60) -> None:
    """Print a formatted header with a title.

    Args:
        title (str): The title of the header
        width (int): The width of the header bar

    """


def print_section(title, width=60) -> None:
    """Print a formatted section header.

    Args:
        title (str): The title of the section
        width (int): The width of the section bar

    """


def display_config(config_dict, width=80) -> None:
    """Display configuration in a clean, structured format.

    Args:
        config_dict (dict): Dictionary containing configuration parameters
        width (int): Width of the display

    """
    print_header("CONFIGURATION", width)

    # Display model information

    print_section("PARAMETERS")

    print_section("EXECUTION SETTINGS")

    print_section("FILE PATHS")


def ensure_directory_exists(directory):
    """Make sure a directory exists, creating it if necessary.

    Args:
        directory (str): Path to directory to ensure exists

    Returns:
        str: Path to the directory

    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        log(f"Created directory: {directory}", "info", VERBOSE_DETAILED)
    return directory
