"""Thread-safe and atomic I/O utilities for Titan trading bot.

Prevents JSON corruption from process crashes, power failures, or
concurrent writes by using write-to-temp + atomic-rename pattern.
"""
import json
import os
import time


def atomic_json_save(filepath: str, data, indent: int = 2):
    """Atomically save JSON data to file.

    Writes to a temporary file first, then uses os.replace() for an
    atomic rename. This prevents corruption if the process crashes
    mid-write or power fails.
    
    On Windows, os.replace() can fail with WinError 5 (Access Denied)
    if the target file is locked by antivirus, VS Code, or another
    process. We retry up to 5 times with back-off, then fall back to
    a direct overwrite as a last resort.
    
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
        
        # Retry os.replace up to 5 times (Windows file locking)
        last_err = None
        for attempt in range(5):
            try:
                os.replace(tmp_path, filepath)
                return  # Success
            except PermissionError as e:
                last_err = e
                time.sleep(0.1 * (attempt + 1))  # 100ms, 200ms, 300ms...
        
        # All retries failed — fall back to direct overwrite
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=indent)
                f.flush()
                os.fsync(f.fileno())
        except Exception:
            raise last_err  # Re-raise original if even fallback fails
        
    except Exception:
        # Clean up temp file on failure
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise
