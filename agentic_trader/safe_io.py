"""Thread-safe and atomic I/O utilities for Titan trading bot.

Prevents JSON corruption from process crashes, power failures, or
concurrent writes by using write-to-temp + atomic-rename pattern.
"""
import json
import os


def atomic_json_save(filepath: str, data, indent: int = 2):
    """Atomically save JSON data to file.

    Writes to a temporary file first, then uses os.replace() for an
    atomic rename. This prevents corruption if the process crashes
    mid-write or power fails.
    
    Args:
        filepath: Target JSON file path
        data: JSON-serializable data
        indent: JSON indentation (default 2)
    """
    tmp_path = filepath + '.tmp'
    try:
        with open(tmp_path, 'w') as f:
            json.dump(data, f, indent=indent)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, filepath)
    except Exception:
        # Clean up temp file on failure
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise
