from __future__ import annotations


def advance_migration_progress(progress: int, *, step: int = 4) -> int:
    """Advance a 0-100 migration clock and stop at arrival."""
    if not 0 <= progress <= 100:
        raise ValueError("progress must be between 0 and 100")
    if step <= 0:
        raise ValueError("step must be positive")
    if progress >= 100:
        return 100
    return min(100, progress + step)
