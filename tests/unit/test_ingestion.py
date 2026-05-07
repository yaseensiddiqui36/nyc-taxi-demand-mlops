"""Unit tests for ingestion helpers."""

from src.data.ingestion import months_to_backfill


def test_months_to_backfill_count():
    months = months_to_backfill(6)
    assert len(months) == 6


def test_months_to_backfill_ascending():
    months = months_to_backfill(4)
    # Each month should be after the previous
    for i in range(1, len(months)):
        y_prev, m_prev = months[i - 1]
        y_cur, m_cur = months[i]
        assert (y_cur, m_cur) > (y_prev, m_prev)


def test_months_to_backfill_valid_range():
    months = months_to_backfill(24)
    for year, month in months:
        assert 1 <= month <= 12
        assert year >= 2020
