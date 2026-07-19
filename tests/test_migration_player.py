import pytest

from core.migration_player import advance_migration_progress


def test_migration_progress_advances_and_stops_at_arrival():
    assert advance_migration_progress(40, step=5) == 45
    assert advance_migration_progress(98, step=5) == 100
    assert advance_migration_progress(100, step=5) == 100


@pytest.mark.parametrize("progress", [-1, 101])
def test_migration_progress_rejects_values_outside_clock(progress):
    with pytest.raises(ValueError):
        advance_migration_progress(progress)


def test_migration_progress_requires_positive_step():
    with pytest.raises(ValueError):
        advance_migration_progress(50, step=0)
